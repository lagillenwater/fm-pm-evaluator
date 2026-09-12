#!/bin/bash
#SBATCH --job-name=rung1-simple-score
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 1, score stage: rung 0's two group-bys and the rung 1 table from the cached tensors
# and generations. Submit with --dependency=afterok:<generation array job id>.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="${RUNG1_SIMPLE_CACHE:-/scratch/alpine/$USER/rung1_simple/cache_v2}"
OUT_DIR="${RUNG1_SIMPLE_OUT:-/scratch/alpine/$USER/rung1_simple/results_v3}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID cache=$CACHE out=$OUT_DIR"

python scripts/rung1_simple_tables.py --stage score \
    --rung0-cache "/scratch/alpine/$USER/rung0_cache_v3b" --tahoe-dir "$TAHOE_DE_DIR" \
    --genelist "$STACK_GENELIST" --cache "$CACHE" --out "$OUT_DIR"
