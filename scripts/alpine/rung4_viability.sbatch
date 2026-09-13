#!/bin/bash
#SBATCH --job-name=rung4-viability
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 4: task 0 is the observed run; tasks 1-10 are the matched permutation null (viability
# shuffled across lines within each drug, seed = task id). rung4_combine.sbatch merges them.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
T="${SLURM_ARRAY_TASK_ID:-0}"
BASE="/scratch/alpine/$USER/rung4_viability"
if [ "$T" = "0" ]; then OUT="$BASE/results"; PERM=""; else OUT="$BASE/perm_$T"; PERM="--permute-seed $T"; fi
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID task=$T out=$OUT"
python scripts/rung4_viability.py \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --embeddings "/scratch/alpine/$USER/rung2_bridge/cache_real/bridge_block_embeddings.npz" \
    --prism-dir "/scratch/alpine/$USER/prism19q4" --gen-dir "/scratch/alpine/$USER/rung1_simple/cache_v3" --genelist "$STACK_GENELIST" \
    --out "$OUT" $PERM
