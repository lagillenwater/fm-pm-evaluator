#!/bin/bash
#SBATCH --job-name=rung4-combine
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
BASE="/scratch/alpine/$USER/rung4_viability"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID"
python scripts/rung4_combine.py --observed "$BASE/results" --perm-dir "$BASE" --out "$BASE/results"
