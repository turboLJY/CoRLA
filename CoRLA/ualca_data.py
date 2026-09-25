"""CoRLA cached targets and masked padding for Slime data partitioning."""

import copy


def convert_samples_to_train_data(args, samples):
    """The CoRLA actor consumes structured cached targets in `rewards`.

    Slime's custom conversion hook bypasses scalar reward normalization. Its
    standard partitioner transports this list unchanged to each FSDP rank.
    Pad with masked copies so every rank executes the same number of forwards.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("empty CoRLA sample batch")
    for sample in samples:
        if not sample.train_metadata or "opd_targets" not in sample.train_metadata:
            raise ValueError("CoRLA requires cached training targets from ualca_rollout")
    dp_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    padding = (-len(samples)) % dp_size
    for _ in range(padding):
        dummy = copy.deepcopy(samples[0])
        dummy.loss_mask = [0] * dummy.response_length
        dummy.train_metadata["weight"] = 0.0
        samples.append(dummy)
    return {"tokens": [s.tokens for s in samples],
            "response_lengths": [s.response_length for s in samples],
            "loss_masks": [s.loss_mask for s in samples],
            "rollout_log_probs": [s.rollout_log_probs for s in samples],
            "rewards": [s.train_metadata for s in samples],
            "raw_reward": [s.get_reward_value(args) for s in samples],
            "truncated": [int(s.status == type(s.status).TRUNCATED) for s in samples],
            "group_indices": [s.group_index for s in samples],
            "sample_indices": [s.index for s in samples]}
