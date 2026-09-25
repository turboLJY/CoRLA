"""CPU integration checks with deterministic stand-ins for external services."""

import asyncio
import importlib.util
import json
import math
import os
import queue
import sys
import subprocess
import tempfile
import threading
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ualca_critic import CriticHeads, HFCritic
from ualca_data import convert_samples_to_train_data
from ualca_loss import token_objective
from ualca_signals import UALCAConfig


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class FakeBaseServer:
    async def _handle_request(self, body, session_id, turn_type, session_done):
        self.last_body = body
        await asyncio.sleep(0)
        turns = self._pending_turn_data.setdefault(session_id, {})
        if turns:
            self._fire_opd_task(session_id, max(turns), turns[max(turns)], body["messages"][-1])
        n = len(turns) + 1
        turns[n] = {"prompt_ids": [n], "tools": None, "has_next_state": False}
        return {"response": {"session_id": session_id}}

    def _flush_pending_record(self, *args):
        return


def load_api_without_services():
    # Exercise the actual session and probability-validation implementation;
    # replace only unavailable HTTP/Slime imports, not the methods under test.
    base = types.ModuleType("openclaw_opd_api_server")
    base.OpenClawOPDAPIServer = FakeBaseServer
    for name in ("_append_hint_to_messages", "_build_hint_judge_messages",
                 "_flatten_message_content", "_normalize_messages_for_template"):
        setattr(base, name, lambda x, *args, **kwargs: x)
    fastapi = types.SimpleNamespace(Header=lambda **kw: None, HTTPException=HTTPError, Request=object)
    spec = importlib.util.spec_from_file_location("ualca_api_under_test", Path(__file__).with_name("ualca_api_server.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"openclaw_opd_api_server": base, "fastapi": fastapi, "httpx": types.ModuleType("httpx")}):
        spec.loader.exec_module(module)
    return module


API = load_api_without_services()


def make_server():
    server = API.UALCAAPIServer.__new__(API.UALCAAPIServer)
    server._pending_turn_data = {}
    server._turn_counts = {}
    server._session_locks = {}
    server._round_sessions = set()
    server._finished_sessions = set()
    server._round_limit = 2
    server._round_id = 0
    server.output_queue = queue.Queue()
    server.submission_enabled = threading.Event()
    server.submission_enabled.set()
    server._encode_messages = lambda messages, tools=None: [90 + len(messages)]
    return server


class APITests(unittest.IsolatedAsyncioTestCase):
    async def test_final_action_is_retained_and_feedback_is_idempotent(self):
        server = make_server()
        server._pending_turn_data["task"] = {
            2: {"prompt_ids": [2], "has_next_state": False},
            1: {"prompt_ids": [1], "has_next_state": True, "transition_ids": [2]},
        }
        payload = {"session_id": "task", "terminal_reward": 1,
                   "messages": [{"role": "tool", "content": "tests passed"}]}
        result = await server.finish_trajectory(payload)
        self.assertEqual(result["turns"], 2)
        trajectory = server.output_queue.get_nowait()
        self.assertEqual(trajectory["states"], [[1], [2], [91]])
        self.assertEqual(trajectory["outcome"], 1)
        self.assertFalse(server._pending_turn_data)
        self.assertEqual((await server.finish_trajectory(payload))["status"], "already_finished")
        self.assertTrue(server.output_queue.empty())

    async def test_terminal_outcome_is_explicit_and_truncation_is_distinct(self):
        server = make_server()
        server._pending_turn_data["task"] = {1: {"prompt_ids": [1], "has_next_state": False}}
        body = {"session_id": "task", "messages": [{"role": "user", "content": "timeout"}]}
        with self.assertRaises(HTTPError):
            await server.finish_trajectory(body)
        with self.assertRaises(HTTPError):
            await server.finish_trajectory({**body, "terminated": "false"})
        await server.finish_trajectory({**body, "terminated": False})
        trajectory = server.output_queue.get_nowait()
        self.assertFalse(trajectory["terminated"])
        self.assertIsNone(trajectory["outcome"])

    async def test_concurrent_requests_for_same_session_stay_ordered(self):
        server = make_server()
        body = {"messages": [{"role": "user", "content": "next"}], "temperature": 0.6}
        await asyncio.gather(*[server._handle_request(body, "task", "main", False) for _ in range(2)])
        turns = server._pending_turn_data["task"]
        self.assertEqual(len(turns), 2)
        self.assertTrue(turns[1]["has_next_state"])
        self.assertEqual(server.last_body["temperature"], 1)

    async def test_round_limit_allows_existing_sessions_to_finish(self):
        server = make_server()
        server._round_limit = 1
        body = {"messages": [{"role": "user", "content": "next"}]}
        await server._handle_request(body, "one", "main", False)
        with self.assertRaises(HTTPError) as error:
            await server._handle_request(body, "two", "main", False)
        self.assertEqual(error.exception.status_code, 503)
        await server._handle_request(body, "one", "main", False)
        with self.assertRaises(RuntimeError):
            server.begin_round(1)

    async def test_scorer_checks_token_identity_and_finite_probabilities(self):
        result = {"meta_info": {"input_token_logprobs": [[None, 9], [-0.2, 1], [-0.3, 2]]}}
        self.assertEqual(API.parse_sampled_log_probs(result, [1, 2]), [-0.2, -0.3])
        for bad in ({}, {"meta_info": {"input_token_logprobs": [[math.nan, 1]]}},
                    {"meta_info": {"input_token_logprobs": [[-0.2, 3]]}}):
            with self.assertRaises(ValueError):
                API.parse_sampled_log_probs(bad, [1])


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 4)
        self.embedding.requires_grad_(False)
        self.adapter = nn.Linear(4, 4)

    def forward(self, input_ids, **kwargs):
        hidden = self.adapter(self.embedding(input_ids).cumsum(1))
        return types.SimpleNamespace(last_hidden_state=hidden)


class ModelIntegrationTests(unittest.TestCase):
    def test_critic_alternates_trainable_heads_and_preserves_frozen_backbone(self):
        torch.manual_seed(0)
        critic = HFCritic.__new__(HFCritic)
        critic.config = UALCAConfig(head_hidden=8, potential_updates=1)
        critic.device = torch.device("cpu")
        critic.max_context = 20
        critic.model = ToyBackbone()
        critic.heads = CriticHeads(4, critic.config)
        critic.adapter_parameters = list(critic.model.adapter.parameters())
        critic.optimizer = torch.optim.AdamW(critic.adapter_parameters + list(critic.heads.parameters()), lr=0.01)
        backbone = critic.model.embedding.weight.detach().clone()
        potential_before = [p.detach().clone() for p in critic.heads.potential.parameters()]
        critic.update_judge([[{"ids": [1, 2], "label": 1.0}]])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(potential_before, critic.heads.potential.parameters())))
        judge_before = [p.detach().clone() for p in critic.heads.judge.parameters()]
        critic.update_potential([{"states": [[1], [1, 2], [1, 2, 3]], "terminated": True,
                                 "outcome": 1, "lower": [-0.1, 0.9], "upper": [0.1, 1.0]}])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(judge_before, critic.heads.judge.parameters())))
        self.assertTrue(torch.equal(backbone, critic.model.embedding.weight))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(potential_before, critic.heads.potential.parameters())))
        mu, sigma = critic.predict([[1, 2]])[0]
        self.assertLessEqual(abs(mu), 1)
        self.assertGreaterEqual(sigma, critic.config.sigma_min)

    def test_checkpointed_policy_loss_matches_full_action_gradient(self):
        cfg = UALCAConfig()
        torch.manual_seed(1)
        full = torch.randn(4, 7, requires_grad=True)
        chunked = full.detach().clone().requires_grad_(True)
        reference = torch.randn(4, 7)
        tokens = torch.tensor([1, 3, 2, 0])
        old = full.detach().log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
        opd = torch.tensor([1.0, -0.2, 0, -1])
        mask = torch.tensor([1, 1, 0, 1])
        args = (0.4, 0.99, cfg, 0.2, 0.28)
        expected = token_objective(full, reference, tokens, old, opd, mask, *args)
        expected.backward()
        pieces = [checkpoint(token_objective, chunked[i:i+2], reference[i:i+2], tokens[i:i+2],
                             old[i:i+2], opd[i:i+2], mask[i:i+2], *args, use_reentrant=False)
                  for i in (0, 2)]
        actual = torch.stack(pieces).sum()
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(chunked.grad, full.grad)

    def test_masked_padding_preserves_cached_signals(self):
        class Status(Enum):
            COMPLETED = "completed"
            TRUNCATED = "truncated"
        class Sample:
            def __init__(self, n):
                self.tokens = list(range(n + 1))
                self.response_length = n
                self.loss_mask = [1] * n
                self.rollout_log_probs = [-0.1] * n
                self.train_metadata = {"advantage": 2, "weight": 0.99, "opd_targets": [0.3] * n,
                                       "num_trajectories": 1, "round": 0}
                self.status = Status.COMPLETED
                self.group_index = self.index = 0
            def get_reward_value(self, args):
                return 2
        samples = [Sample(2), Sample(3)]
        args = types.SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=4)
        data = convert_samples_to_train_data(args, samples)
        self.assertEqual(len(data["tokens"]), 4)
        self.assertEqual(data["rewards"][0]["opd_targets"], [0.3, 0.3])
        self.assertEqual(data["loss_masks"][-1], [0, 0])
        self.assertEqual(data["rewards"][-1]["weight"], 0)
        self.assertEqual(samples[0].train_metadata["weight"], 0.99)


class LauncherTests(unittest.TestCase):
    def test_launcher_forwards_configuration_and_selects_corla_driver(self):
        script = Path(__file__).with_name("run_qwen3_4b_openclaw_ualca.sh").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fake_ray = path / "ray"
            fake_ray.write_text(f"#!{sys.executable}\n" +
                "import json, os, sys\nfrom pathlib import Path\n"
                "if sys.argv[1] == 'status': sys.exit(0)\n"
                "args = sys.argv[1:]\n"
                "env = json.loads(Path(args[args.index('--runtime-env') + 1]).read_text())\n"
                "Path(os.environ['CORLA_TEST_CAPTURE']).write_text(json.dumps({'args': args, 'env': env}))\n")
            fake_ray.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCLAW_")}
            secret = 'test-only key with "quotes" and a newline\n'
            capture = path / "capture.json"
            env.update({"PATH": tmp + os.pathsep + env["PATH"], "HF_CKPT": "/test/model with spaces",
                        "OPENCLAW_UALCA_BOOTSTRAP_PATH": "/test/bootstrap.jsonl",
                        "OPENCLAW_UALCA_LAMBDA_A": "0.7", "SGLANG_API_KEY": secret,
                        "CORLA_TEST_CAPTURE": str(capture), "USE_WANDB": "0"})
            completed = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(capture.read_text())
            self.assertIn(str(script.with_name("train_corla.py")), result["args"])
            self.assertIn("--disable-rollout-trim-samples", result["args"])
            self.assertIn("/test/model with spaces", result["args"])
            self.assertEqual(result["env"]["env_vars"]["OPENCLAW_UALCA_LAMBDA_A"], "0.7")
            self.assertEqual(result["env"]["env_vars"]["SGLANG_API_KEY"], secret)
            self.assertNotIn(secret, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
