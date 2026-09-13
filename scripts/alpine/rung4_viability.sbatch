#!/bin/bash
#SBATCH --job-name=rung4-viability
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 4: PRISM primary repurposing viability (public, downloaded once to scratch) for the Tahoe
# lines x drugs; the line-specific response predicted from the line's baseline and from its
# Tahoe expression response, against a replicate split-half ceiling.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID"
python scripts/rung4_viability.py \
    --tensors "/scratch/alpine/$USER/rung1_simple/cache_v2/tensors.npz" \
    --embeddings "/scratch/alpine/$USER/rung2_bridge/cache_real/bridge_block_embeddings.npz" \
    --prism-dir "/scratch/alpine/$USER/prism19q4" --gen-dir "/scratch/alpine/$USER/rung1_simple/cache_v3" --genelist "$STACK_GENELIST" \
    --out "/scratch/alpine/$USER/rung4_viability/results"
