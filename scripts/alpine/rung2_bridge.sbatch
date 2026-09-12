#!/bin/bash
#SBATCH --job-name=rung2-bridge
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 2 on the real cells from tahoe_real_cells: every line's baseline (512 real DMSO cells)
# and every (line, drug) treated state with >= 128 real cells, each against its synthetic
# counterpart, embedded with the base Stack encoder. Submit after the combine job.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="${RUNG2_CACHE:-/scratch/alpine/$USER/rung2_bridge/cache_real}"
OUT_DIR="${RUNG2_OUT:-/scratch/alpine/$USER/rung2_bridge/results_real}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID cache=$CACHE out=$OUT_DIR"

python scripts/rung2_bridge.py --stage "${STAGE:-all}" \
    --real-cells "/scratch/alpine/$USER/tahoe_real_cells/cache/real_cells_5um.h5ad" \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --genelist "$STACK_GENELIST" \
    --cache "$CACHE" --out "$OUT_DIR"
