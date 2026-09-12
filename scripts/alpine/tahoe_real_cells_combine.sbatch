#!/bin/bash
#SBATCH --job-name=tahoe-real-cells-combine
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
# Merge the 16 blocks of tahoe_real_cells.sbatch into real_cells_5um.h5ad and the count table.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="${REAL_CELLS_CACHE:-/scratch/alpine/$USER/tahoe_real_cells/cache}"
OUT_DIR="${REAL_CELLS_OUT:-/scratch/alpine/$USER/tahoe_real_cells/results}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID cache=$CACHE out=$OUT_DIR"

python scripts/tahoe_real_cells.py \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --tahoe-dir "$TAHOE_DE_DIR" \
    --cache "$CACHE" --out "$OUT_DIR" \
    --n-blocks 16 --combine
