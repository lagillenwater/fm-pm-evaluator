#!/bin/bash
#SBATCH --job-name=tahoe-real-cells
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G                    # 4 cores x 3,840 MB
#SBATCH --time=10:00:00
#SBATCH --array=0-15%8               # 16 blocks of the 3,388 shards, 8 readers at a time
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Real Tahoe cells for rung 1's block: DMSO cells of the 49 lines (512 per line) and treated
# cells at 5 uM for the 108 drugs (128 per line and drug), from the raw expression shards.
# Authenticated through the token in ~/.git-credentials (rung1_env.sh). Combine afterwards:
#   ralpine submit scripts/alpine/tahoe_real_cells_combine.sbatch --dependency=afterok:<array id>
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh
if [ -z "${HF_TOKEN:-}" ]; then echo "no Hugging Face token found (HF_TOKEN empty); see rung1_env.sh" >&2; exit 2; fi
python -c "import pyarrow, huggingface_hub" 2>/dev/null || pip install -q "pyarrow>=15" "huggingface_hub>=0.23"

CACHE="${REAL_CELLS_CACHE:-/scratch/alpine/$USER/tahoe_real_cells/cache}"
OUT_DIR="${REAL_CELLS_OUT:-/scratch/alpine/$USER/tahoe_real_cells/results}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID block=${SLURM_ARRAY_TASK_ID:-0} cache=$CACHE out=$OUT_DIR"

python scripts/tahoe_real_cells.py \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --tahoe-dir "$TAHOE_DE_DIR" \
    --cache "$CACHE" --out "$OUT_DIR" \
    --n-blocks 16 --block "${SLURM_ARRAY_TASK_ID:-0}"
