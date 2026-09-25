"""Synchronous driver: collect -> critic/OPD -> policy -> synchronize.

The existing train_async.py overlaps rounds. CoRLA requires every cached target
and hint-conditioned score to use one frozen rollout-policy version instead.
"""

import ray

from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args):
    import slime.backends.fsdp_utils as fsdp
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from ualca_actor import CoRLAFSDPActor

    if args.train_backend != "fsdp" or not args.use_lora:
        raise ValueError("train_corla.py requires the FSDP backend and policy LoRA")
    if args.colocate or args.offload_rollout or args.offload_train:
        raise ValueError("this CoRLA driver uses separate policy, rollout, PRM, and critic GPUs")
    if args.rollout_temperature != 1.0 or args.rollout_top_p != 1.0:
        raise ValueError("CoRLA requires temperature=1 and top_p=1 for on-policy probabilities")
    if args.use_kl_loss or args.kl_coef != 0 or args.ref_update_interval is not None:
        raise ValueError("CoRLA applies its own exact forward KL to the frozen initial policy")
    if not args.disable_rollout_trim_samples:
        raise ValueError("CoRLA requires --disable-rollout-trim-samples to preserve full trajectories")
    # RayTrainGroup resolves this exported implementation when it constructs
    # actors. Restrict the override to construction within this driver.
    original_actor = fsdp.FSDPTrainRayActor
    fsdp.FSDPTrainRayActor = CoRLAFSDPActor
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, rounds_per_epoch = create_rollout_manager(args, pgs["rollout"], pgs.get("prm"))
    try:
        actor, critic, prm_teacher = create_training_models(args, pgs, rollout_manager)
    finally:
        fsdp.FSDPTrainRayActor = original_actor
    if critic is not None or prm_teacher is not None:
        raise ValueError("CoRLA manages its own critic; a Megatron distillation teacher is not used")
    actor.update_weights()
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        data = ray.get(rollout_manager.generate.remote(rollout_id))
        ray.get(actor.async_train(rollout_id, data))
        if should_run_periodic_action(rollout_id, args.save_interval, rounds_per_epoch, args.num_rollout):
            actor.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
        actor.update_weights()
        if should_run_periodic_action(rollout_id, args.eval_interval, rounds_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))
    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    train(parse_args())
