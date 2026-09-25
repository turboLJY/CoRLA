"""FSDP policy LoRA actor for cached CoRLA targets (Eq. 18).

Only this method's driver installs the subclass. The standard Slime FSDP actor
does not consume custom losses or per-token OPD targets, so inheriting its PPO
training loop would silently omit part of the method.
"""

import logging

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from slime.backends.fsdp_utils.actor import FSDPTrainRayActor
from slime.utils import logging_utils
from ualca_loss import token_objective
from ualca_signals import UALCAConfig

logger = logging.getLogger(__name__)


class CoRLAFSDPActor(FSDPTrainRayActor):
    def _train_core(self, rollout_id, rollout_data):
        if not self._is_lora:
            raise ValueError("CoRLA policy updates require --use-lora")
        cfg = UALCAConfig.from_env()
        device = torch.device("cuda", torch.cuda.current_device())
        # Keep gradient checkpointing active, but disable dropout for consistent
        # pi_old / pi_current probabilities during the policy update.
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = cfg.lr * min((rollout_id + 1) / cfg.warmup_rounds, 1.0)
        reported_loss = torch.zeros((), device=device)
        count = None
        for ids, n, old, mask, target in zip(
            rollout_data["tokens"], rollout_data["response_lengths"],
            rollout_data["rollout_log_probs"], rollout_data["loss_masks"], rollout_data["rewards"], strict=True
        ):
            if target["round"] != rollout_id:
                raise ValueError("cached CoRLA targets belong to a different rollout-policy version")
            if n < 1 or len(old) != n or len(target["opd_targets"]) != n:
                raise ValueError("misaligned action tokens, old probabilities, or OPD targets")
            if count is not None and count != target["num_trajectories"]:
                raise ValueError("inconsistent trajectory normalization")
            count = target["num_trajectories"]
            tokens = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens),
                      "use_cache": False, "logits_to_keep": n + 1}
            # With a frozen backbone, disabling the policy adapter exactly
            # recovers the initial policy. Cache reference logits on CPU to
            # avoid a second vocabulary-sized GPU allocation during training.
            reference = None
            if cfg.kl_coef:
                with torch.no_grad(), self.model.disable_adapter():
                    reference = self.model(**inputs).logits[0, -n - 1:-1].detach().cpu()
            logits = self.model(**inputs).logits[0, -n - 1:-1]
            old_lp = torch.as_tensor(old, device=device, dtype=torch.float32)
            opd = torch.as_tensor(target["opd_targets"], device=device, dtype=torch.float32)
            masks = torch.as_tensor(mask, device=device, dtype=torch.float32)
            response_tokens = tokens[0, -n:]
            losses = []
            # Checkpoint vocab softmaxes so peak loss memory is one chunk;
            # action-token sums and trajectory normalization are unchanged.
            for start in range(0, n, 128):
                end = min(start + 128, n)
                ref_chunk = reference[start:end] if reference is not None else None
                losses.append(checkpoint(
                    token_objective, logits[start:end], ref_chunk, response_tokens[start:end],
                    old_lp[start:end], opd[start:end], masks[start:end], target["advantage"],
                    target["weight"], cfg, self.args.eps_clip, self.args.eps_clip_high,
                    use_reentrant=False))
            loss = torch.stack(losses).sum()
            # FSDP averages gradients over DP ranks. Cancel that average and
            # divide by trajectories, not turns, tokens, or padded samples.
            (loss * self.dp_size / count).backward()
            reported_loss += loss.detach()
            del logits, reference, losses, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(reported_loss, group=self.dp_group)
        if dist.get_rank() == 0:
            metrics = {"train/loss": float(reported_loss / count), "train/step": rollout_id,
                       "train/grad_norm": float(grad_norm), "train/lr": self.optimizer.param_groups[0]["lr"]}
            logger.info("[CoRLA] %s", metrics)
            logging_utils.log(self.args, metrics, step_key="train/step")
        self.global_step += 1
        self.prof.step(rollout_id=rollout_id)
