#!/bin/bash
#SBATCH --job-name=rung3-platform
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 3: LINCS 2020 Level 5 compound signatures (public, 35.5 GB, downloaded to scratch once)
# -> one mean landmark signature per (LINCS line, Tahoe drug) at 24 h / ~10 uM -> the L1000 drug
# mean predicting each Tahoe line's response, against the Tahoe drug mean and a shuffled-drug
# null, on the own-DE genes within the 978 landmarks.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh
python -c "import h5py" 2>/dev/null || pip install -q "h5py>=3"

L1000_DIR="/scratch/alpine/$USER/lincs2020"
CACHE="/scratch/alpine/$USER/rung3_platform/cache"
OUT_DIR="/scratch/alpine/$USER/rung3_platform/results"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID l1000=$L1000_DIR out=$OUT_DIR"

python scripts/rung3_platform.py --stage "${STAGE:-all}" \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --l1000-dir "$L1000_DIR" --cache "$CACHE" --out "$OUT_DIR"
