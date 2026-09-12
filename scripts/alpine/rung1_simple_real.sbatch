#!/bin/bash
#SBATCH --job-name=rung1-real-gen
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1                  # 0 stack_cytokine, 1 stack_sciplex
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 1 Stack column on REAL cells: prompt and query are real DMSO cells, context real
# treated cells at 5 uM, from tahoe_real_cells. Tensors reused from cache_v2; generations go
# to cache_v3 so the synthetic-cell arrays are kept. Score with
#   ralpine submit scripts/alpine/rung1_simple_real_score.sbatch --dependency=afterok:<array id>
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="/scratch/alpine/$USER/rung1_simple/cache_v3"
OUT_DIR="/scratch/alpine/$USER/rung1_simple/results_v4_real"
MODELS=(stack_cytokine stack_sciplex)
GEN_MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID model=$GEN_MODEL cache=$CACHE out=$OUT_DIR"

python scripts/rung1_simple_tables.py --stage generate --gen-model "$GEN_MODEL" --gen-batch 2 \
    --real-cells "/scratch/alpine/$USER/tahoe_real_cells/cache/real_cells_5um.h5ad" \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --rung0-cache "/scratch/alpine/$USER/rung0_cache_v3b" --tahoe-dir "$TAHOE_DE_DIR" \
    --genelist "$STACK_GENELIST" --cache "$CACHE" --out "$OUT_DIR"
