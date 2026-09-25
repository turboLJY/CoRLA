"""Synchronous CoRLA rounds connected to the OpenClaw proxy and Slime."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import time
from pathlib import Path

from fastapi import HTTPException
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import eval_rollout
from slime.utils.async_utils import run
from slime.utils.types import Sample
from ualca_api_server import UALCAAPIServer
from ualca_data import convert_samples_to_train_data
from ualca_signals import UALCAConfig
from ualca_training import CoRLATrainer, RayCriticBackend

logger = logging.getLogger(__name__)
_global_worker = None


class CoRLAWorker:
    def __init__(self, args):
        import threading
        self.args = args
        self.config = UALCAConfig.from_env()
        self.output_queue = queue.Queue()
        self.enabled = threading.Event()
        self.server = UALCAAPIServer(args, self.output_queue, self.enabled)
        self.backend = RayCriticBackend(args.hf_checkpoint, self.config, args.rollout_max_context_len)
        self.trainer = CoRLATrainer(self.backend, self.config)
        self.initialized = False
        self.server.start()

    async def server_call(self, method, *args):
        while self.server._loop is None:
            if not self.server._thread.is_alive():
                raise RuntimeError("CoRLA proxy failed to start; check the listen port and server logs")
            await asyncio.sleep(0.05)

        async def invoke():
            result = getattr(self.server, method)(*args)
            return await result if asyncio.iscoroutine(result) else result

        future = asyncio.run_coroutine_threadsafe(invoke(), self.server._loop)
        return await asyncio.wrap_future(future)

    async def initialize(self, rollout_id):
        if self.initialized:
            return
        resume = os.getenv("OPENCLAW_UALCA_RESUME")
        if resume:
            await self.trainer.load(resume)
        else:
            bootstrap = os.getenv("OPENCLAW_UALCA_BOOTSTRAP_PATH")
            if not bootstrap:
                raise ValueError("set OPENCLAW_UALCA_BOOTSTRAP_PATH to initial-policy teacher-labeled transitions")
            with open(bootstrap) as stream:
                pairs = [json.loads(line) for line in stream if line.strip()]
            metrics = await self.trainer.bootstrap(pairs)
            logger.info("[CoRLA] bootstrap completed: %s", metrics)
        if self.trainer.round_id != rollout_id:
            raise ValueError(f"policy starts at round {rollout_id}, critic expects {self.trainer.round_id}; load matching checkpoints")
        self.initialized = True

    async def collect(self, rollout_id):
        await self.server_call("begin_round", rollout_id)
        trajectories = []
        last_log = time.monotonic()
        while len(trajectories) < self.args.rollout_batch_size:
            try:
                trajectories.append(self.output_queue.get_nowait())
            except queue.Empty:
                await asyncio.sleep(0.05)
            if time.monotonic() - last_log >= 30:
                logger.info("[CoRLA] waiting for terminal feedback: %d/%d trajectories",
                            len(trajectories), self.args.rollout_batch_size)
                last_log = time.monotonic()
        self.enabled.clear()
        return trajectories

    async def generate(self, rollout_id):
        await self.initialize(rollout_id)
        trajectories = await self.collect(rollout_id)

        async def label(transition):
            return await self.server_call("query_teacher_label", transition)

        signals, metrics = await self.trainer.run_round(trajectories, label)
        groups = []
        selected_count = 0
        for trajectory, shaped in zip(trajectories, signals, strict=True):
            group = []
            group_id = next(self.server._group_counter)
            for t, (transition, signal) in enumerate(zip(trajectory["transitions"], shaped, strict=True)):
                response_ids = transition["response_ids"]
                old_lp = await self.server_call("score_policy", transition["prompt_ids"], response_ids)
                targets = [0.0] * len(response_ids)
                hint = None
                if signal.selective_opd and self.config.lambda_opd > 0:
                    hint = await self.server_call("query_hint", transition)
                    if hint:
                        try:
                            hinted_ids = await self.server_call("hinted_prompt_ids", transition, hint)
                            if len(hinted_ids) + len(response_ids) > self.args.rollout_max_context_len:
                                raise ValueError("hint-conditioned prompt exceeds context limit")
                        except (ValueError, HTTPException) as exc:
                            # Only context validation is optional. Scorer failures
                            # below abort the round rather than fabricating targets.
                            logger.warning("[CoRLA] hint skipped: %s", exc)
                            hint = None
                        if hint:
                            hinted_lp = await self.server_call("score_policy", hinted_ids, response_ids)
                            targets = [max(-self.config.opd_clip, min(self.config.opd_clip, h - o))
                                       for h, o in zip(hinted_lp, old_lp, strict=True)]
                            selected_count += 1
                sample = Sample()
                sample.prompt = transition["prompt_text"]
                sample.response = transition["response_text"]
                sample.tokens = list(transition["prompt_ids"]) + list(response_ids)
                sample.response_length = len(response_ids)
                sample.loss_mask = [1] * len(response_ids)
                sample.rollout_log_probs = old_lp
                sample.status = Sample.Status.COMPLETED if trajectory["terminated"] else Sample.Status.TRUNCATED
                sample.index = next(self.server._index_counter)
                sample.group_index = group_id
                sample.reward = {"score": signal.advantage}
                sample.metadata["ualca"] = {**signal.to_metadata(), "opd_mask": bool(hint),
                                            "session_id": trajectory["session_id"], "turn": t + 1}
                sample.train_metadata = {"advantage": signal.advantage, "weight": self.config.gamma ** t,
                                         "opd_targets": targets, "num_trajectories": len(trajectories),
                                         "round": rollout_id}
                self.server._append_prm_record({"session_id": trajectory["session_id"], "turn": t + 1,
                                               "hint": hint, "ualca": sample.metadata["ualca"]})
                group.append(sample)
            groups.append(group)
        metrics["ualca/opd_selected"] = selected_count
        outcomes = [t["outcome"] for t in trajectories if t["terminated"]]
        if outcomes:
            metrics["ualca/terminal_reward"] = sum(outcomes) / len(outcomes)
        # Save on the same rounds as policy checkpoints. The driver saves the
        # policy only after this round's update has succeeded.
        interval = self.args.save_interval
        if self.args.save and ((interval and (rollout_id + 1) % interval == 0) or rollout_id + 1 == self.args.num_rollout):
            await self.trainer.save(str(Path(self.args.save) / "corla" / f"round_{rollout_id:06d}"))
        return RolloutFnTrainOutput(samples=groups, metrics=metrics)

    def stop(self):
        self.enabled.clear()
        self.server.stop()
        self.backend.close()




def generate_rollout_openclaw_ualca(args, rollout_id, data_buffer, evaluation=False):
    global _global_worker
    if evaluation:
        result, _ = run(eval_rollout(args, rollout_id))
        return result
    if _global_worker is None:
        _global_worker = CoRLAWorker(args)
    return run(_global_worker.generate(rollout_id))


def stop_global_worker():
    global _global_worker
    if _global_worker is not None:
        _global_worker.stop()
        _global_worker = None


atexit.register(stop_global_worker)
