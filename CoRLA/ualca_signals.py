"""CoRLA equations (8)--(16) in ICLR2027.pdf, without model dependencies."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class UALCAConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    lambda_a: float = 0.95
    lambda_cp: float = 0.1
    beta_phi: float = 1.0
    reward_radius: float = 1.0
    tau_opd: float = 0.5
    opd_clip: float = 2.0
    lambda_opd: float = 0.1
    kl_coef: float = 0.01
    sigma_min: float = 1e-3
    huber_delta: float = 1.0
    calibration_window: int = 2048
    calibration_rounds: int = 32
    replay_window: int = 10000
    judge_queries: int = 64
    calibration_queries: int = 64
    judge_updates: int = 5
    potential_updates: int = 5
    bootstrap_updates: int = 200
    bootstrap_calibration: int = 1024
    batch_size: int = 128
    head_hidden: int = 1024
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-6
    warmup_rounds: int = 20
    seed: int = 42

    def __post_init__(self):
        if not 0 < self.alpha < 1 or not 0 < self.gamma <= 1:
            raise ValueError("alpha must be in (0,1), gamma in (0,1]")
        if not 0 <= self.lambda_a <= 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid trace decay or LoRA dropout")
        positive = ("beta_phi", "reward_radius", "opd_clip", "sigma_min", "huber_delta", "lr")
        nonnegative = ("lambda_cp", "tau_opd", "lambda_opd", "kl_coef")
        for name in positive + nonnegative:
            v = getattr(self, name)
            if not math.isfinite(v) or (v <= 0 if name in positive else v < 0):
                raise ValueError(f"invalid {name}: {v}")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{f.name} must be finite")
            if isinstance(v, int) and f.name != "seed" and v < 1:
                raise ValueError(f"{f.name} must be positive")

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(**{
            f.name: type(getattr(defaults, f.name))(
                os.getenv(f"OPENCLAW_UALCA_{f.name.upper()}", str(getattr(defaults, f.name)))
            ) for f in fields(cls)
        })


@dataclass(frozen=True)
class RewardInference:
    mu: float
    sigma: float
    half_width: float
    lower: float
    upper: float
    gate: float


@dataclass(frozen=True)
class ShapedSignal:
    reward: RewardInference
    env_reward: float
    potential: float
    next_potential: float
    potential_difference: float
    credit: float
    advantage: float
    selective_opd: bool

    def to_metadata(self):
        # Cold-start intervals stay infinite in memory; JSON logs mark them.
        data = {**asdict(self.reward), **asdict(self)}
        data.pop("reward")
        data["unbounded_interval"] = math.isinf(self.reward.half_width)
        return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in data.items()}


def conformal_quantile(scores: list[float], alpha: float) -> float:
    """Finite-sample rank, including the +infinity sentinel (Eq. 8)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if any(not math.isfinite(s) or s < 0 for s in scores):
        raise ValueError("calibration residuals must be finite and nonnegative")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return sorted(scores)[rank - 1] if rank <= len(scores) else math.inf


def infer_reliable_reward(mu: float, sigma: float, quantile: float, config: UALCAConfig):
    """Use the UNCLIPPED interval width. No vote or mean-dependent gate."""
    if not math.isfinite(mu) or abs(mu) > 1 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("judge must return mu in [-1,1] and a positive finite sigma")
    if math.isnan(quantile) or quantile < 0:
        raise ValueError("invalid conformal quantile")
    b = quantile * sigma
    return RewardInference(mu, sigma, b, mu - b, mu + b,
                           1.0 - min(b / config.reward_radius, 1.0))


def terminal_returns(length: int, outcome: float, gamma: float) -> list[float]:
    if (length < 1 or not isinstance(outcome, (int, float))
            or not math.isfinite(outcome) or not 0 <= outcome <= 1):
        raise ValueError("a nonempty trajectory and verified outcome in [0,1] are required")
    return [gamma ** (length - 1 - t) * outcome for t in range(length)]


def shape_trajectory(rewards: list[RewardInference], potentials: list[float],
                     outcome: float | None, terminated: bool, config: UALCAConfig):
    """Eqs. 13--14. Only true termination forces Phi(T+1)=0.

    Truncations retain the final state's learned bootstrap potential and have
    no fabricated terminal outcome. No second value baseline is subtracted.
    """
    n = len(rewards)
    if n < 1 or len(potentials) != n + 1 or not all(map(math.isfinite, potentials)):
        raise ValueError("expected T rewards and T+1 finite potential predictions")
    if terminated:
        terminal_returns(n, outcome, config.gamma)
    elif outcome is not None:
        raise ValueError("truncations cannot carry a terminal outcome")
    phi = list(potentials)
    if terminated:
        phi[-1] = 0.0
    env = [0.0] * n
    if terminated:
        env[-1] = float(outcome)
    differences = [config.gamma * phi[t + 1] - phi[t] for t in range(n)]
    credits = [env[t] + config.beta_phi * rewards[t].gate * differences[t] for t in range(n)]
    advantages = [0.0] * n
    trace = 0.0
    for t in reversed(range(n)):
        trace = credits[t] + config.gamma * config.lambda_a * trace
        advantages[t] = trace
    return [ShapedSignal(rewards[t], env[t], phi[t], phi[t + 1], differences[t],
                         credits[t], advantages[t], rewards[t].half_width <= config.tau_opd)
            for t in range(n)]
