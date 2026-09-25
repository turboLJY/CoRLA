"""Algorithm 1: teacher querying, disjoint calibration, and alternating updates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from ualca_signals import UALCAConfig, conformal_quantile, infer_reliable_reward, shape_trajectory, terminal_returns


def input_key(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class RayCriticBackend:
    def __init__(self, model_path, config, max_context=32768):
        import ray
        from ualca_critic import HFCritic
        self.actor = ray.remote(num_gpus=1)(HFCritic).remote(model_path, config, max_context=max_context)

    async def call(self, name, *args):
        import ray
        ref = getattr(self.actor, name).remote(*args)
        return await asyncio.to_thread(ray.get, ref)

    def close(self):
        import ray
        ray.kill(self.actor)


class CoRLATrainer:
    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config or UALCAConfig.from_env()
        self.rng = random.Random(self.config.seed)
        self.replay = []
        self.probes = []
        self.train_keys = set()
        self.calibration_keys = set()
        self.quantile = math.inf
        self.bootstrapped = False
        self.round_id = 0

    def _pair(self, ids, label, round_id):
        if (not ids or not all(type(i) is int and i >= 0 for i in ids)
                or not isinstance(label, (int, float)) or not math.isfinite(label) or abs(label) > 1):
            raise ValueError("teacher pairs require nonempty token IDs and finite labels in [-1,1]")
        return {"ids": list(ids), "label": float(label), "round": round_id, "key": input_key(ids)}

    async def bootstrap(self, pairs):
        unique = {}
        for pair in pairs:
            p = self._pair(pair["ids"], pair["label"], -1)
            if p["key"] in unique and p["label"] != unique[p["key"]]["label"]:
                raise ValueError("bootstrap contains conflicting labels for the same transition")
            unique[p["key"]] = p
        data = list(unique.values())
        self.rng.shuffle(data)
        ncal = self.config.bootstrap_calibration
        if len(data) <= ncal:
            raise ValueError(f"bootstrap needs more than {ncal} distinct labeled transitions")
        self.probes = data[:ncal][-self.config.calibration_window:]
        self.calibration_keys = {p["key"] for p in data[:ncal]}
        train = data[ncal:]
        self.train_keys = {p["key"] for p in train}
        self.replay = train[-self.config.replay_window:]
        batches = [self.rng.choices(train, k=self.config.batch_size)
                   for _ in range(self.config.bootstrap_updates)]
        loss = await self.backend.call("update_judge", batches)
        await self._recalibrate()
        self.bootstrapped = True
        return {"bootstrap/judge_loss": loss, "bootstrap/train": len(train), "bootstrap/calibration": len(self.probes)}

    async def _recalibrate(self):
        predictions = await self.backend.call("predict", [p["ids"] for p in self.probes], "judge")
        scores = [abs(p["label"] - mu) / sigma
                  for p, (mu, sigma) in zip(self.probes, predictions, strict=True)]
        self.quantile = conformal_quantile(scores, self.config.alpha)

    def select_queries(self, transitions, predictions):
        """Top-K sigma + uniform probes from the remainder, with no data leakage."""
        unique = {}
        for i, t in enumerate(transitions):
            unique.setdefault(input_key(t["transition_ids"]), i)
        candidates = list(unique.values())
        eligible = [i for i in candidates if input_key(transitions[i]["transition_ids"]) not in self.calibration_keys]
        eligible.sort(key=lambda i: predictions[i][1], reverse=True)
        judge = eligible[:min(self.config.judge_queries, len(candidates) // 2)]
        remaining = [i for i in candidates if i not in judge
                     and input_key(transitions[i]["transition_ids"]) not in self.train_keys]
        probes = self.rng.sample(remaining, min(self.config.calibration_queries, len(remaining)))
        return judge, probes

    async def run_round(self, trajectories, query_teacher):
        """query_teacher(transition) returns one frozen-teacher label or None.

        Each trajectory carries T transitions, T+1 prefix token lists, and an
        explicit terminal/truncated marker. Hint extraction happens afterwards.
        """
        if not self.bootstrapped:
            raise RuntimeError("bootstrap the judge using a separate initial-policy dataset first")
        if not trajectories or any(not t["transitions"] for t in trajectories):
            raise ValueError("empty trajectory batch")
        for trajectory in trajectories:
            n = len(trajectory["transitions"])
            if len(trajectory["states"]) != n + 1 or type(trajectory["terminated"]) is not bool:
                raise ValueError("expected T+1 states and an explicit boolean termination marker")
            if trajectory["terminated"]:
                terminal_returns(n, trajectory["outcome"], self.config.gamma)
            elif trajectory["outcome"] is not None:
                raise ValueError("truncations must not carry a fabricated terminal outcome")
        transitions = [t for trajectory in trajectories for t in trajectory["transitions"]]
        ids = [t["transition_ids"] for t in transitions]
        before = await self.backend.call("predict", ids, "judge")
        judge_indices, probe_indices = self.select_queries(transitions, before)
        # Cache coverage predictions BEFORE labels arrive or the judge changes.
        coverage_intervals = {i: infer_reliable_reward(*before[i], self.quantile, self.config)
                              for i in probe_indices}
        selected = judge_indices + probe_indices
        labels = await asyncio.gather(*[query_teacher(transitions[i]) for i in selected])
        labeled = {i: self._pair(ids[i], y, self.round_id)
                   for i, y in zip(selected, labels, strict=True) if y is not None}
        fresh = [labeled[i] for i in judge_indices if i in labeled]
        historical = list(self.replay)
        self.replay = (self.replay + fresh)[-self.config.replay_window:]
        self.train_keys.update(p["key"] for p in fresh)
        await self.backend.call("set_round", self.round_id)
        batches = []
        for _ in range(self.config.judge_updates):
            nnew = self.config.batch_size // 2 if fresh else 0
            batch = self.rng.choices(fresh, k=nnew) if nnew else []
            batch += self.rng.choices(historical or self.replay, k=self.config.batch_size - nnew)
            batches.append(batch)
        judge_loss = await self.backend.call("update_judge", batches)
        new_probes = [labeled[i] for i in probe_indices if i in labeled]
        self.calibration_keys.update(p["key"] for p in new_probes)
        self.probes = [p for p in self.probes + new_probes
                       if p["round"] > self.round_id - self.config.calibration_rounds]
        self.probes = self.probes[-self.config.calibration_window:]
        await self._recalibrate()
        updated = await self.backend.call("predict", ids, "judge")
        # These immutable floats are reused after updating the shared adapter.
        cached = [infer_reliable_reward(mu, sigma, self.quantile, self.config) for mu, sigma in updated]
        potential_data = []
        offset = 0
        for trajectory in trajectories:
            n = len(trajectory["transitions"])
            intervals = cached[offset:offset + n]
            potential_data.append({"states": trajectory["states"], "outcome": trajectory["outcome"],
                                   "terminated": trajectory["terminated"],
                                   "lower": [r.lower for r in intervals], "upper": [r.upper for r in intervals]})
            offset += n
        phi_loss = await self.backend.call("update_potential", potential_data)
        signals = []
        offset = 0
        for trajectory in trajectories:
            n = len(trajectory["transitions"])
            potentials = await self.backend.call("predict", trajectory["states"], "potential")
            signals.append(shape_trajectory(cached[offset:offset + n], potentials,
                                            trajectory["outcome"], trajectory["terminated"], self.config))
            offset += n
        covered = [coverage_intervals[i].lower <= labeled[i]["label"] <= coverage_intervals[i].upper
                   for i in probe_indices if i in labeled]
        metrics = {"ualca/judge_loss": judge_loss, "ualca/potential_loss": phi_loss,
                   "ualca/judge_labels": len(fresh), "ualca/calibration_labels": len(new_probes),
                   "ualca/teacher_failures": len(selected) - len(labeled),
                   "ualca/calibration_size": len(self.probes),
                   "ualca/unbounded_quantile": float(math.isinf(self.quantile)),
                   "ualca/gate": sum(x.gate for x in cached) / len(cached)}
        if math.isfinite(self.quantile):
            metrics["ualca/quantile"] = self.quantile
        if covered:
            metrics["ualca/pre_update_teacher_coverage"] = sum(covered) / len(covered)
        self.round_id += 1
        return signals, metrics

    async def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        await self.backend.call("save", str(path))
        state = {"config": asdict(self.config), "round_id": self.round_id,
                 "quantile": self.quantile if math.isfinite(self.quantile) else None,
                 "replay": self.replay, "probes": self.probes, "train_keys": sorted(self.train_keys),
                 "calibration_keys": sorted(self.calibration_keys), "rng": self.rng.getstate()}
        (path / "trainer.json").write_text(json.dumps(state, allow_nan=False))

    async def load(self, directory):
        path = Path(directory)
        state = json.loads((path / "trainer.json").read_text())
        if state["config"] != asdict(self.config):
            raise ValueError("critic checkpoint configuration differs from this run")
        await self.backend.call("load", str(path))
        self.replay, self.probes = state["replay"], state["probes"]
        self.train_keys, self.calibration_keys = set(state["train_keys"]), set(state["calibration_keys"])
        self.round_id = state["round_id"]
        self.quantile = math.inf if state["quantile"] is None else state["quantile"]
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        self.bootstrapped = True
