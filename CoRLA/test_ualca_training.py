import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from ualca_signals import UALCAConfig
from ualca_training import CoRLATrainer, input_key


class FakeBackend:
    def __init__(self):
        self.events = []
        self.mean = 0.0

    async def call(self, name, *args):
        self.events.append((name, copy.deepcopy(args)))
        if name == "predict":
            ids, kind = args
            if kind == "judge":
                return [(self.mean, 0.2 + (x[0] % 5) / 10) for x in ids]
            return [x[0] / 100 for x in ids]
        if name == "update_judge":
            self.mean = 0.1
            return 0.5
        if name == "update_potential":
            self.mean = 0.9  # Shared critic changed: cached intervals must survive.
            return 0.25
        if name == "save":
            Path(args[0], "fake.json").write_text(json.dumps({"mean": self.mean}))
        if name == "load":
            self.mean = json.loads(Path(args[0], "fake.json").read_text())["mean"]


def trajectory(offset=100, terminated=True):
    transitions = [{"transition_ids": [offset + i]} for i in range(3)]
    return {"transitions": transitions, "states": [[offset - 1], [offset], [offset + 1], [offset + 2]],
            "outcome": 1 if terminated else None, "terminated": terminated}


class TrainerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = UALCAConfig(alpha=0.25, bootstrap_calibration=3, bootstrap_updates=1,
                               judge_updates=1, potential_updates=1, batch_size=4,
                               judge_queries=1, calibration_queries=1)
        self.backend = FakeBackend()
        self.trainer = CoRLATrainer(self.backend, self.cfg)
        await self.trainer.bootstrap([{"ids": [i], "label": 0.2} for i in range(12)])

    async def test_bootstrap_split_is_disjoint(self):
        self.assertFalse(self.trainer.train_keys & self.trainer.calibration_keys)
        batches = next(args[0] for name, args in self.backend.events if name == "update_judge")
        seen = {p["key"] for b in batches for p in b}
        self.assertFalse(seen & self.trainer.calibration_keys)

    async def test_query_selection_uses_sigma_and_uniform_disjoint_probes(self):
        ts = [{"transition_ids": [100 + i]} for i in range(4)]
        judge, probes = self.trainer.select_queries(ts, [(0, x) for x in [0.1, 0.8, 0.3, 0.2]])
        self.assertEqual(judge, [1])
        self.assertFalse(set(judge) & set(probes))

    async def test_round_order_caches_intervals_before_potential_changes_judge(self):
        self.backend.events.clear()
        async def teacher(t):
            return 0.2
        signals, metrics = await self.trainer.run_round([trajectory()], teacher)
        events = [name for name, _ in self.backend.events]
        self.assertLess(events.index("update_judge"), events.index("update_potential"))
        potential_pos = events.index("update_potential")
        self.assertTrue(all(name != "predict" or args[1] == "potential"
                            for name, args in self.backend.events[potential_pos + 1:]))
        self.assertAlmostEqual(signals[0][0].reward.mu, 0.1)
        self.assertEqual(self.backend.mean, 0.9)
        self.assertIn("ualca/pre_update_teacher_coverage", metrics)
        self.assertFalse(self.trainer.train_keys & self.trainer.calibration_keys)

    async def test_recalibration_rescores_stored_inputs(self):
        probes = list(self.trainer.probes)
        self.backend.mean = 0.8
        await self.trainer._recalibrate()
        scores = sorted(abs(p["label"] - 0.8) / (0.2 + p["ids"][0] % 5 / 10) for p in probes)
        self.assertAlmostEqual(self.trainer.quantile, scores[-1])

    async def test_query_failures_do_not_become_neutral_labels(self):
        replay = len(self.trainer.replay)
        probes = len(self.trainer.probes)
        async def failing_teacher(t):
            return None
        _, metrics = await self.trainer.run_round([trajectory()], failing_teacher)
        self.assertEqual(len(self.trainer.replay), replay)
        self.assertEqual(len(self.trainer.probes), probes)
        self.assertEqual(metrics["ualca/teacher_failures"], 2)

    async def test_replay_mixes_new_with_historical_labels(self):
        historical = set(self.trainer.train_keys)
        async def teacher(t):
            return 0.2
        await self.trainer.run_round([trajectory()], teacher)
        batches = [args[0] for name, args in self.backend.events if name == "update_judge"][-1]
        batch = batches[0]
        self.assertEqual(sum(p["key"] in historical for p in batch), 2)
        self.assertEqual(len(batch), 4)

    async def test_repeated_probe_never_enters_judge_regression(self):
        probe_ids = self.trainer.probes[0]["ids"]
        ts = [{"transition_ids": probe_ids}, {"transition_ids": [999]}]
        judge, _ = self.trainer.select_queries(ts, [(0, 1000), (0, 1)])
        self.assertEqual(judge, [1])

    async def test_checkpoint_preserves_round_buffers_quantile_and_rng(self):
        async def teacher(t):
            return 0.2
        await self.trainer.run_round([trajectory()], teacher)
        with tempfile.TemporaryDirectory() as tmp:
            await self.trainer.save(tmp)
            restored = CoRLATrainer(FakeBackend(), self.cfg)
            await restored.load(tmp)
            self.assertEqual(restored.round_id, 1)
            self.assertEqual(restored.probes, self.trainer.probes)
            self.assertEqual(restored.replay, self.trainer.replay)
            self.assertEqual(restored.quantile, self.trainer.quantile)
            self.assertEqual(restored.rng.random(), self.trainer.rng.random())


if __name__ == "__main__":
    unittest.main()
