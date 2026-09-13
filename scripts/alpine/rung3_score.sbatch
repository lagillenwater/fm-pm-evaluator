#!/bin/bash
#SBATCH --job-name=rung3-score
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID"
python scripts/rung3_platform.py --stage score \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --l1000-dir "/scratch/alpine/$USER/lincs2020" --genelist "$STACK_GENELIST" \
    --rung1-cache "/scratch/alpine/$USER/rung1_simple/cache_v3" \
    --cache "/scratch/alpine/$USER/rung3_platform/cache" --out "/scratch/alpine/$USER/rung3_platform/results"
