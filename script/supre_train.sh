#!/usr/bin/env bash
# Train DM05 on the converted Supre LeRobot data.
# Override DATA_DIR, MODEL_DIR, NPROC_PER_NODE, NUM_TRAIN_STEPS, or OUTPUT_DIR
# when launching this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-opendm-supre}"

CONDA_BASE="$(conda info --base)"
# Make the script work from a non-interactive shell and under nohup.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/supre_pickup_long_opendm}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/checkpoints/DM05}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/user_checkpoints/dm05_supre}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"

if [[ ! -d "$DATA_DIR/jsonl" ]]; then
    echo "Converted Supre JSONL data not found: $DATA_DIR/jsonl" >&2
    exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "DM05 checkpoint not found: $MODEL_DIR" >&2
    echo "Download it with: hf download Dexmal/DM05 --local-dir $MODEL_DIR" >&2
    exit 1
fi

export OPENDM_SUPRE_JSONL_DIR="$DATA_DIR/jsonl"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$REPO_ROOT/script/dm05_launcher.sh" \
    --exp "$REPO_ROOT/playground/dm05_supre.py" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --task train \
    --data-config.dataset-name supre_pickup_long \
    --model-config.model-name-or-path "$MODEL_DIR" \
    --model-config.chunk-size "$CHUNK_SIZE" \
    --model-config.vision-attn-implementation sdpa \
    --model-config.llm-attn-implementation flex_attention \
    --model-config.action-attn-implementation sdpa \
    --trainer-config.output-dir "$OUTPUT_DIR" \
    --trainer-config.num-train-steps "$NUM_TRAIN_STEPS"
