"""Frozen HF backbone + critic LoRA + independent judge/potential MLPs.

The policy is trained by ualca_actor on a separate copy of the same initial
backbone. Physical separation keeps policy and critic optimizers independent.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ualca_loss import location_scale_loss, potential_loss
from ualca_signals import UALCAConfig, terminal_returns


class CriticHeads(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.sigma_min = config.sigma_min
        self.judge = nn.Sequential(nn.Linear(hidden_size, config.head_hidden), nn.SiLU(),
                                   nn.Linear(config.head_hidden, 2))
        self.potential = nn.Sequential(nn.Linear(hidden_size, config.head_hidden), nn.SiLU(),
                                       nn.Linear(config.head_hidden, 1))

    def predict_judge(self, hidden):
        mean, scale = self.judge(hidden.float()).unbind(-1)
        return mean.tanh(), (self.sigma_min ** 2 + F.softplus(scale)).sqrt()


class HFCritic:
    """Ray actor implementation; calls are serialized on its reserved GPU."""

    def __init__(self, model_path, config=None, device="cuda", max_context=32768):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        self.config = config or UALCAConfig.from_env()
        self.device = torch.device(device)
        self.max_context = max_context
        self.rng = random.Random(self.config.seed)
        torch.manual_seed(self.config.seed)
        backbone = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            attn_implementation="sdpa")
        self.model = get_peft_model(backbone, LoraConfig(
            r=self.config.lora_rank, lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )).to(self.device)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.heads = CriticHeads(backbone.config.hidden_size, self.config).to(self.device)
        self.adapter_parameters = [p for p in self.model.parameters() if p.requires_grad]
        # A single optimizer preserves the shared critic adapter's Adam moments
        # across the alternating judge and potential stages.
        self.optimizer = torch.optim.AdamW(
            self.adapter_parameters + list(self.heads.parameters()), lr=self.config.lr,
            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    def _hidden(self, ids):
        if not ids or len(ids) > self.max_context:
            raise ValueError("critic context is empty or exceeds max_context; history is never silently truncated")
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state[0, -1]

    def _stage(self, judge):
        self.model.train()
        self.heads.train()
        self.heads.judge.requires_grad_(judge)
        self.heads.potential.requires_grad_(not judge)

    def _step(self):
        torch.nn.utils.clip_grad_norm_(self.adapter_parameters + list(self.heads.parameters()), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def set_round(self, round_id):
        scale = min((round_id + 1) / self.config.warmup_rounds, 1.0)
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.lr * scale

    @torch.no_grad()
    def predict(self, token_sequences, kind="judge"):
        self.model.eval()
        self.heads.eval()
        if kind == "judge":
            return [tuple(float(x) for x in self.heads.predict_judge(self._hidden(ids)))
                    for ids in token_sequences]
        if kind != "potential":
            raise ValueError(f"unknown prediction kind: {kind}")
        return [float(self.heads.potential(self._hidden(ids).float()).squeeze()) for ids in token_sequences]

    def update_judge(self, batches):
        self._stage(judge=True)
        losses = []
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for pair in batch:
                mu, sigma = self.heads.predict_judge(self._hidden(pair["ids"]))
                loss = location_scale_loss(mu, sigma, mu.new_tensor(pair["label"]), self.config.huber_delta)
                (loss / len(batch)).backward()
                total += float(loss.detach()) / len(batch)
            self._step()
            losses.append(total)
        return sum(losses) / max(len(losses), 1)

    def update_potential(self, trajectories):
        self._stage(judge=False)
        losses = []
        for _ in range(self.config.potential_updates):
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            # Accumulate one transition at a time to bound activation memory.
            # Summing over each complete trajectory preserves Eq. 10--11's
            # trajectory average despite variable numbers of transitions.
            for trajectory in trajectories:
                states = trajectory["states"]
                terminated = trajectory["terminated"]
                n = len(states) - 1
                targets = terminal_returns(n, trajectory["outcome"], self.config.gamma) if terminated else None
                for t in range(n):
                    phi = self.heads.potential(self._hidden(states[t]).float()).squeeze()
                    terminal_step = terminated and t == n - 1
                    next_phi = phi * 0 if terminal_step else self.heads.potential(self._hidden(states[t + 1]).float()).squeeze()
                    values = torch.stack([phi, next_phi])
                    target = None if targets is None else phi.new_tensor([targets[t]])
                    loss, _, _ = potential_loss(
                        values, target, phi.new_tensor([trajectory["lower"][t]]),
                        phi.new_tensor([trajectory["upper"][t]]), self.config.gamma,
                        self.config.lambda_cp, self.config.huber_delta, terminated=terminal_step)
                    (loss / len(trajectories)).backward()
                    total += float(loss.detach()) / len(trajectories)
            self._step()
            losses.append(total)
        return sum(losses) / max(len(losses), 1)

    def save(self, directory):
        from peft import get_peft_model_state_dict
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"adapter": get_peft_model_state_dict(self.model), "heads": self.heads.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "rng": self.rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}, path / "critic.pt")

    def load(self, directory):
        from peft import set_peft_model_state_dict
        state = torch.load(Path(directory) / "critic.pt", map_location=self.device, weights_only=False)
        set_peft_model_state_dict(self.model, state["adapter"])
        self.heads.load_state_dict(state["heads"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
