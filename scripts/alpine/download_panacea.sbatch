#!/bin/bash
#SBATCH --job-name=panacea-download
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
python -c "import synapseclient, synapseutils" 2>/dev/null || pip install -q "synapseclient>=4"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID"
python scripts/download_panacea.py --out "/scratch/alpine/$USER/panacea"
