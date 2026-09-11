#!/bin/bash
#SBATCH --job-name=rung1-descriptions
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G                    # 8 cores x 3,840 MB; PCA/NMF over 50 lines x 15,012
                                     # genes and 107 drugs' fingerprints, no table scan
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 1, stage 4: the non-Stack line descriptions (expression, PCA, NMF, random stand-ins),
# drug fingerprints and Tanimoto similarity, and the descriptions' build known-answer control.
# Reads task 4's cache (expression.parquet, expression_halves.parquet), so it waits on the DMSO
# cells combine job, not on the embeddings (it does not touch Stack).
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_JOB_ID:-} cache=$RUNG1_CACHE out_dir=$OUT_DIR"

python scripts/heldout_descriptions.py \
    --grid "$OUT_DIR/rung1_grid.json" \
    --drug-metadata "$RUNG1_CACHE/metadata/drug_metadata.parquet" \
    --cache "$RUNG1_CACHE" \
    --out-dir "$OUT_DIR"
