"""Differentiable CoRLA objectives, Eqs. (5), (10)--(11), (16)--(18)."""

import torch
import torch.nn.functional as F


def location_scale_loss(mu, sigma, labels, delta=1.0):
    residual_loss = F.huber_loss(mu, labels, delta=delta, reduction="none")
    variance = sigma.square()
    return (residual_loss / (2 * variance) + 0.5 * variance.log()).mean()


def potential_loss(potentials, returns, lower, upper, gamma=0.99, lambda_cp=0.1,
                   delta=1.0, terminated=True):
    """One trajectory's sum; the caller averages over trajectories.

    On a truncation, returns=None: no unobserved outcome is used as a label.
    Its final increment is still eligible for the interval constraint.
    """
    zero = potentials.sum() * 0.0
    ret_loss = zero if returns is None else F.huber_loss(
        potentials[:-1], returns.detach(), delta=delta, reduction="sum")
    differences = gamma * potentials[1:] - potentials[:-1]
    stop = len(differences) - int(terminated)
    lo, hi = lower[:stop].detach(), upper[:stop].detach()
    diff = differences[:stop]
    cp_loss = (F.relu(lo - diff).square() + F.relu(diff - hi).square()).sum()
    return ret_loss + lambda_cp * cp_loss, ret_loss, cp_loss


def opd_targets(old_log_probs, hinted_log_probs, selected, clip=2.0):
    if not selected:
        return torch.zeros_like(old_log_probs)
    if hinted_log_probs is None or hinted_log_probs.shape != old_log_probs.shape:
        raise ValueError("selected OPD requires aligned sampled-token probabilities")
    return (hinted_log_probs.detach() - old_log_probs.detach()).clamp(-clip, clip)


def clipped_surrogate(new_log_probs, old_log_probs, advantages, eps_lo=0.2, eps_hi=0.28):
    ratio = (new_log_probs - old_log_probs.detach()).clamp(-20, 20).exp()
    advantage = torch.as_tensor(advantages, device=ratio.device, dtype=ratio.dtype).detach()
    return torch.minimum(ratio * advantage, ratio.clamp(1 - eps_lo, 1 + eps_hi) * advantage)


def policy_loss(new_log_probs, old_log_probs, advantage, opd_advantages, mask,
                weight=1.0, lambda_opd=0.1, eps_lo=0.2, eps_hi=0.28):
    """Token SUM with temporal weighting; trajectory averaging is external.

    Keep the two clipped terms separate even when their advantages disagree.
    Unselected steps have exactly zero cached OPD targets.
    """
    rl = clipped_surrogate(new_log_probs, old_log_probs, advantage, eps_lo, eps_hi)
    opd = clipped_surrogate(new_log_probs, old_log_probs, opd_advantages, eps_lo, eps_hi)
    mask = mask.detach().to(new_log_probs)
    return -float(weight) * ((rl + lambda_opd * opd) * mask).sum()


def forward_kl(logits, reference_logits):
    """Exact KL(pi_current || pi_initial), over the complete vocabulary."""
    log_p = logits.float().log_softmax(-1)
    log_q = reference_logits.detach().to(logits.device).float().log_softmax(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def token_objective(logits, reference_logits, tokens, old_lp, opd, mask, advantage,
                    weight, config, eps_lo, eps_hi):
    log_probs = logits.float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    loss = policy_loss(log_probs, old_lp, advantage, opd, mask, weight,
                       config.lambda_opd, eps_lo, eps_hi)
    if config.kl_coef:
        loss = loss + config.kl_coef * weight * (forward_kl(logits, reference_logits) * mask).sum()
    return loss
