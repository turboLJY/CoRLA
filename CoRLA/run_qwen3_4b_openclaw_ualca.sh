#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
: "${HF_CKPT:?Set HF_CKPT to the initial Qwen3-4B Hugging Face checkpoint}"
if [[ -z "${OPENCLAW_UALCA_RESUME:-}" ]]; then
    : "${OPENCLAW_UALCA_BOOTSTRAP_PATH:?Set a JSONL file containing initial-policy transition ids and teacher labels}"
fi

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
PRM_GPUS=${PRM_GPUS:-1}
PRM_TP=${PRM_TP:-1}
CRITIC_GPUS=1
if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS + CRITIC_GPUS > NUM_GPUS )); then
    echo "CoRLA needs ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS + 1 critic GPU <= NUM_GPUS" >&2
    exit 1
fi
SAVE_CKPT="${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen3-4b-corla}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
export SGLANG_API_KEY="${SGLANG_API_KEY:-}"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${OPENCLAW_RECORD_FILE:-${SCRIPT_DIR}/results/corla_record.jsonl}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"
export OPENCLAW_UALCA_BOOTSTRAP_PATH="${OPENCLAW_UALCA_BOOTSTRAP_PATH:-}"
export OPENCLAW_UALCA_RESUME="${OPENCLAW_UALCA_RESUME:-}"
export PRM_M=1
export OPENCLAW_EVAL_MODE=0

# Defaults live in UALCAConfig, so local and remote actors use the same values.
# Forward all explicit overrides, including paths, via a JSON file. Never print
# credentials or interpolate them into a shell command or hand-built JSON.
RUNTIME_ENV_FILE="$(mktemp /tmp/corla-runtime-XXXXXX.json)"
trap 'rm -f "$RUNTIME_ENV_FILE"' EXIT
python - "$RUNTIME_ENV_FILE" <<'PY'
import json, os, sys
from pathlib import Path
from ualca_signals import UALCAConfig
UALCAConfig.from_env()  # fail early on malformed settings
names = {"PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER", "HOST", "PORT",
         "SERVED_MODEL_NAME", "SGLANG_API_KEY", "PRM_M", "WANDB_API_KEY"}
env = {k: v for k, v in os.environ.items() if k in names or k.startswith("OPENCLAW_")}
Path(sys.argv[1]).write_text(json.dumps({"env_vars": env}))
PY

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if ! ray status >/dev/null 2>&1; then
    ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

LOAD_ARGS=()
if [[ -n "${LOAD_CKPT:-}" ]]; then
    LOAD_ARGS=(--load "$LOAD_CKPT")
fi
WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == 1 ]]; then
    WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-openclaw_rl}" --wandb-group qwen3-4b-corla)
fi

ray job submit --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    --runtime-env "$RUNTIME_ENV_FILE" -- \
    python "$SCRIPT_DIR/train_corla.py" \
    --train-backend fsdp \
    --actor-num-nodes 1 --actor-num-gpus-per-node "$ACTOR_GPUS" \
    --rollout-num-gpus "$ROLLOUT_GPUS" --num-gpus-per-node "$NUM_GPUS" \
    --hf-checkpoint "$HF_CKPT" --save "$SAVE_CKPT" --save-interval "${SAVE_INTERVAL:-100}" \
    "${LOAD_ARGS[@]}" \
    --disable-rollout-global-dataset --disable-rollout-trim-samples \
    --rollout-function-path ualca_rollout.generate_rollout_openclaw_ualca \
    --custom-convert-samples-to-train-data-path ualca_rollout.convert_samples_to_train_data \
    --num-rollout "${NUM_ROLLOUT:-500}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}" \
    --seed "${OPENCLAW_UALCA_SEED:-42}" \
    --n-samples-per-prompt 1 --num-steps-per-rollout 1 \
    --rollout-max-response-len 4096 --rollout-max-context-len 32768 \
    --rollout-temperature 1.0 --rollout-top-p 1.0 \
    --reward-key score --disable-rewards-normalization --use-rollout-logprobs \
    --eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0 \
    --use-lora --lora-rank "${OPENCLAW_UALCA_LORA_RANK:-16}" \
    --lora-alpha "${OPENCLAW_UALCA_LORA_ALPHA:-32}" \
    --lora-dropout "${OPENCLAW_UALCA_LORA_DROPOUT:-0.05}" \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --optimizer adam --lr "${OPENCLAW_UALCA_LR:-1e-6}" --lr-decay-style constant \
    --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --clip-grad 1.0 \
    --gradient-checkpointing --attn-implementation sdpa \
    --use-dynamic-batch-size --max-tokens-per-gpu 32768 \
    --rollout-num-gpus-per-engine "$ROLLOUT_TP" \
    --sglang-mem-fraction-static 0.8 --sglang-context-length 32768 \
    --sglang-reasoning-parser qwen3 --sglang-tool-call-parser "${TOOL_CALL_PARSER:-qwen25}" \
    --prm-enable --prm-num-gpus "$PRM_GPUS" --prm-num-gpus-per-engine "$PRM_TP" \
    --prm-model-path "$PRM_MODEL_PATH" --prm-m 1 --prm-temperature 0 --prm-max-new-tokens 256 \
    "${WANDB_ARGS[@]}" "$@"
