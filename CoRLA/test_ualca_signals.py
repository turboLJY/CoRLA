import math
import unittest

import torch

from ualca_loss import location_scale_loss, opd_targets, policy_loss, potential_loss, forward_kl
from ualca_signals import UALCAConfig, conformal_quantile, infer_reliable_reward, shape_trajectory, terminal_returns


class SignalTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_infinity_sentinel(self):
        self.assertEqual(conformal_quantile([], 0.1), math.inf)
        self.assertEqual(conformal_quantile([0.0] * 8, 0.1), math.inf)
        self.assertEqual(conformal_quantile(list(range(9)), 0.1), 8)

    def test_interval_is_not_clipped_to_label_range(self):
        reward = infer_reliable_reward(0.95, 1.2, 1, UALCAConfig())
        self.assertAlmostEqual(reward.upper, 2.15)
        self.assertEqual(reward.gate, 0)

    def test_gate_is_linear_and_independent_of_mean(self):
        cfg = UALCAConfig()
        for mean in (-0.9, 0, 0.9):
            reward = infer_reliable_reward(mean, 0.2, 2, cfg)
            self.assertAlmostEqual(reward.gate, 0.6)

    def test_infinite_interval_disables_credit_increment_and_opd(self):
        cfg = UALCAConfig()
        reward = infer_reliable_reward(1, 0.1, math.inf, cfg)
        signal = shape_trajectory([reward], [10, 20], 1, True, cfg)[0]
        self.assertEqual(signal.credit, 1)
        self.assertFalse(signal.selective_opd)
        self.assertTrue(signal.to_metadata()["unbounded_interval"])
        self.assertIsNone(signal.to_metadata()["upper"])

    def test_precise_zero_mean_can_trigger_opd_inclusively(self):
        cfg = UALCAConfig(tau_opd=0.5)
        reward = infer_reliable_reward(0, 0.5, 1, cfg)
        self.assertTrue(shape_trajectory([reward], [0, 0], 0, True, cfg)[0].selective_opd)

    def test_terminal_credit_preserves_outcome_and_masks_last_potential(self):
        cfg = UALCAConfig(gamma=1, lambda_a=1)
        rewards = [infer_reliable_reward(0, 1, 0, cfg)] * 2
        signals = shape_trajectory(rewards, [0.1, 0.4, 100], 1, True, cfg)
        self.assertAlmostEqual(signals[0].credit, 0.3)
        self.assertAlmostEqual(signals[1].credit, 0.6)
        self.assertAlmostEqual(signals[0].advantage, 0.9)
        self.assertEqual(signals[-1].next_potential, 0)

    def test_truncation_bootstraps_without_fabricated_outcome(self):
        cfg = UALCAConfig(gamma=1)
        reward = infer_reliable_reward(0, 1, 0, cfg)
        signal = shape_trajectory([reward], [0.2, 0.8], None, False, cfg)[0]
        self.assertAlmostEqual(signal.credit, 0.6)
        self.assertEqual(signal.next_potential, 0.8)
        with self.assertRaises(ValueError):
            shape_trajectory([reward], [0.2, 0.8], 0, False, cfg)

    def test_late_outcome_changes_early_advantage(self):
        cfg = UALCAConfig(gamma=0.8, lambda_a=0.5)
        rewards = [infer_reliable_reward(0, 2, 1, cfg)] * 3
        success = shape_trajectory(rewards, [0] * 4, 1, True, cfg)
        failure = shape_trajectory(rewards, [0] * 4, 0, True, cfg)
        self.assertAlmostEqual(success[0].advantage, 0.16)
        self.assertEqual(failure[0].advantage, 0)

    def test_returns_are_terminal_outcomes_not_local_judge_means(self):
        self.assertEqual(terminal_returns(3, 1, 0.5), [0.25, 0.5, 1.0])
        self.assertEqual(terminal_returns(3, 0, 0.5), [0, 0, 0])
        with self.assertRaises(ValueError):
            terminal_returns(3, None, 0.5)

    def test_configuration_rejects_invalid_scales(self):
        for kwargs in ({"alpha": 0}, {"sigma_min": 0}, {"tau_opd": math.inf}, {"lambda_a": 2}):
            with self.assertRaises(ValueError):
                UALCAConfig(**kwargs)


class LossTests(unittest.TestCase):
    def test_location_scale_matches_equation_five(self):
        loss = location_scale_loss(torch.tensor(0.0), torch.tensor(2.0), torch.tensor(1.0))
        self.assertAlmostEqual(float(loss), 0.5 / 8 + math.log(4) / 2, places=6)

    def test_interval_penalty_excludes_true_terminal_and_detaches_endpoints(self):
        phi = torch.tensor([0.2, 0.4, 0.0], requires_grad=True)
        lower = torch.tensor([0.1, 0.9], requires_grad=True)
        upper = torch.tensor([0.3, 1.0], requires_grad=True)
        loss, _, cp = potential_loss(phi, torch.tensor([1.0, 1.0]), lower, upper, gamma=1)
        self.assertEqual(float(cp.detach()), 0)
        loss.backward()
        self.assertIsNone(lower.grad)
        self.assertIsNone(upper.grad)
        self.assertIsNotNone(phi.grad)

    def test_wider_intervals_impose_weaker_constraints(self):
        phi = torch.tensor([0.0, 0.8, 0.0])
        _, _, tight = potential_loss(phi, None, torch.tensor([0.1, 0.0]), torch.tensor([0.2, 0.0]), gamma=1)
        _, _, wide = potential_loss(phi, None, torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]), gamma=1)
        self.assertGreater(float(tight), float(wide))

    def test_truncated_last_increment_is_constrained(self):
        _, _, cp = potential_loss(torch.tensor([0.2, 0.8]), None, torch.tensor([0.0]),
                                 torch.tensor([0.1]), gamma=1, terminated=False)
        self.assertAlmostEqual(float(cp), 0.25)

    def test_opd_mask_is_exactly_zero_and_targets_are_detached(self):
        old = torch.tensor([-2.0, -1.0], requires_grad=True)
        hinted = torch.tensor([-0.1, -9.0], requires_grad=True)
        self.assertTrue(torch.equal(opd_targets(old, None, False), torch.zeros(2)))
        targets = opd_targets(old, hinted, True, clip=1)
        self.assertTrue(torch.equal(targets, torch.tensor([1.0, -1.0])))
        self.assertFalse(targets.requires_grad)

    def test_rl_and_opd_are_clipped_separately(self):
        new = torch.tensor([math.log(1.5)], requires_grad=True)
        loss = policy_loss(new, torch.zeros(1), 1.0, torch.tensor([-1.0]), torch.ones(1), lambda_opd=1)
        self.assertAlmostEqual(float(loss.detach()), 0.22, places=6)
        loss.backward()
        self.assertGreater(float(new.grad), 0)

    def test_token_sum_temporal_weight_and_observation_mask(self):
        loss = policy_loss(torch.zeros(3), torch.zeros(3), 2.0, torch.zeros(3),
                           torch.tensor([1, 0, 1]), weight=0.5)
        self.assertEqual(float(loss), -2)

    def test_forward_kl_uses_full_distribution_and_frozen_reference(self):
        logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
        reference = torch.zeros_like(logits, requires_grad=True)
        kl = forward_kl(logits, reference).sum()
        self.assertGreater(float(kl.detach()), 0)
        kl.backward()
        self.assertIsNone(reference.grad)
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
