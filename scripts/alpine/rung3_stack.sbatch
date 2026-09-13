#!/bin/bash
#SBATCH --job-name=rung3-stack
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 3, Stack on the pure LINCS set: prompt = L1000 vehicle pseudo-cells, context = L1000
# treated pseudo-cells of the drug, query = real Tahoe DMSO cells of the held-out line.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
MODELS=(stack_cytokine stack_sciplex); GEN_MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID model=$GEN_MODEL"
python scripts/rung3_platform.py --stage stack --gen-model "$GEN_MODEL" \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --real-cells "/scratch/alpine/$USER/tahoe_real_cells/cache/real_cells_5um.h5ad" \
    --l1000-dir "/scratch/alpine/$USER/lincs2020" --genelist "$STACK_GENELIST" \
    --cache "/scratch/alpine/$USER/rung3_platform/cache" --out "/scratch/alpine/$USER/rung3_platform/results"
