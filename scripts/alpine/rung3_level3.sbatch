#!/bin/bash
#SBATCH --job-name=rung3-level3
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 3, LINCS side: LINCS 2020 Level 3 (treated 89 GB + vehicle 19 GB, public, to scratch
# once), instances of the Tahoe drugs at 24 h / ~10 uM in every LINCS line with their plate
# vehicles; per (line, drug) log2FC and per-line baseline over the landmarks; the instance
# matrices kept for the Stack pseudo-cells.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
python -c "import h5py" 2>/dev/null || pip install -q "h5py>=3"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID"
python scripts/rung3_platform.py --stage level3 \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --l1000-dir "/scratch/alpine/$USER/lincs2020" \
    --cache "/scratch/alpine/$USER/rung3_platform/cache" --out "/scratch/alpine/$USER/rung3_platform/results"
