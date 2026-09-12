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
# Rung 2, the bridge test: stream 256 real DMSO cells per line from Tahoe-100M (public, no
# token), embed real halves and library-matched synthetic cells with the base Stack encoder,
# and rank each synthetic population against every line's real population. One job; results
# on scratch; nothing committed.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh
# rung1_env.sh exports HF_TOKEN from a token file that does not exist on the compute nodes,
# which leaves an EMPTY token in the environment and huggingface_hub then sends the header
# "Authorization: Bearer " -- the exact failure of job 32425977. Tahoe-100M is public.
if [ -z "${HF_TOKEN:-}" ]; then unset HF_TOKEN; fi
python -c "import pyarrow, huggingface_hub" 2>/dev/null || pip install -q "pyarrow>=15" "huggingface_hub>=0.23"

CACHE="${RUNG2_CACHE:-/scratch/alpine/$USER/rung2_bridge/cache}"
OUT_DIR="${RUNG2_OUT:-/scratch/alpine/$USER/rung2_bridge/results}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID cache=$CACHE out=$OUT_DIR"

python scripts/rung2_bridge.py --stage "${STAGE:-all}" \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --tahoe-dir "$TAHOE_DE_DIR" \
    --genelist "$STACK_GENELIST" \
    --cache "$CACHE" \
    --out "$OUT_DIR"
