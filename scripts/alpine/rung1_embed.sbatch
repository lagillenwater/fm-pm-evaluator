#!/bin/bash
#SBATCH --job-name=rung1-embed
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # GPU-node memory, not acpu's per-core billing; one
                                     # checkpoint load plus up to 50,000 cells' embeddings
#SBATCH --time=04:00:00
#SBATCH --array=0-2                  # 0 base, 1 cytokine, 2 drug
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 1, stage 3: a 3-task GPU array, one Stack checkpoint load per task, embedding every grid
# line's cached DMSO cells (task 4's cells/line_{i}.h5ad). ah200 (H200) is this stage's
# partition; fall back to --partition=aa100 --qos=gpu-normal --gres=gpu:a100:1 if ah200's queue
# is deep when this is submitted (design.md section 9: August's Stack jobs moved the other way,
# off aa100's long backlog, but aa100 remains the fallback).
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

case "$SLURM_ARRAY_TASK_ID" in
    0) VERSION=base;     CHECKPOINT="$CKPT_BASE";     BASE_CHECKPOINT_ARGS=() ;;
    1) VERSION=cytokine; CHECKPOINT="$CKPT_CYTOKINE";  BASE_CHECKPOINT_ARGS=() ;;
    2) VERSION=drug;     CHECKPOINT="$CKPT_DRUG";      BASE_CHECKPOINT_ARGS=(--base-checkpoint "$CKPT_BASE") ;;
    *) echo "unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (want 0, 1 or 2)" >&2; exit 1 ;;
esac

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-} version=$VERSION checkpoint=$CHECKPOINT cache=$RUNG1_CACHE"

python scripts/heldout_embed.py \
    --version "$VERSION" \
    --checkpoint "$CHECKPOINT" \
    --genelist "$STACK_GENELIST" \
    --grid "$OUT_DIR/rung1_grid.json" \
    --cache "$RUNG1_CACHE" \
    --out-dir "$OUT_DIR" \
    "${BASE_CHECKPOINT_ARGS[@]}"
