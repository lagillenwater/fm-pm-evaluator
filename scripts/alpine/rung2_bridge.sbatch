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
# Rung 2, the bridge test: 256 real DMSO cells per line from the context shards already on the
# checkout (context_by_drug/), real halves and library-matched synthetic cells embedded with the base Stack encoder,
# and rank each synthetic population against every line's real population. One job; results
# on scratch; nothing committed.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="${RUNG2_CACHE:-/scratch/alpine/$USER/rung2_bridge/cache}"
OUT_DIR="${RUNG2_OUT:-/scratch/alpine/$USER/rung2_bridge/results}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID cache=$CACHE out=$OUT_DIR"

python scripts/rung2_bridge.py --stage "${STAGE:-all}" \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --genelist "$STACK_GENELIST" \
    --cache "$CACHE" \
    --out "$OUT_DIR"
