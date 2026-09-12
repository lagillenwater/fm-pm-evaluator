#!/bin/bash
#SBATCH --job-name=rung1-real-score
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="/scratch/alpine/$USER/rung1_simple/cache_v3"
OUT_DIR="/scratch/alpine/$USER/rung1_simple/results_v4_real"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID cache=$CACHE out=$OUT_DIR"

python scripts/rung1_simple_tables.py --stage score \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --rung0-cache "/scratch/alpine/$USER/rung0_cache_v3b" --tahoe-dir "$TAHOE_DE_DIR" \
    --genelist "$STACK_GENELIST" --cache "$CACHE" --out "$OUT_DIR"
