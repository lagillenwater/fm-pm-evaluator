#!/bin/bash
#SBATCH --job-name=rung3-baseline-sim
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID"
python scripts/rung3_baseline_similarity.py \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --l1000 "/scratch/alpine/$USER/rung3_platform/cache/rung3_l1000.npz" \
    --crosswalk docs/tasks/rung1-held-out-prediction/rung1_line_crosswalk.csv \
    --real-cells "/scratch/alpine/$USER/tahoe_real_cells/cache/real_cells_5um.h5ad" \
    --out "/scratch/alpine/$USER/rung3_platform/results"
