#!/bin/bash
#SBATCH --job-name=rung1-dmso-cells
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G                    # 4 cores x 3,840 MB; each task reads a handful of
                                     # Tahoe-100M cell-matrix shards, not the full 3,388
#SBATCH --time=06:00:00              # Hugging Face shard reads, throttled by %4 below
#SBATCH --array=0-63%4               # N_DMSO_BLOCKS=64 blocks, DMSO_CONCURRENCY=4 at once --
                                     # rung 0 saw many concurrent HF readers get rate-limited
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 1, stage 2a: one job-array task per block of Tahoe-100M's cell-matrix shards, keeping
# DMSO cells of the grid's 50 lines. --array here MUST match N_DMSO_BLOCKS (rung1_env.sh); the
# combine job reads the same count. A task whose block is already done skips, so a failed index
# is resubmitted alone with --array=<index>.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} block=${SLURM_ARRAY_TASK_ID:-} n_blocks=$N_DMSO_BLOCKS cache=$RUNG1_CACHE"

python scripts/heldout_dmso_cells.py \
    --grid "$OUT_DIR/rung1_grid.json" \
    --crosswalk "$OUT_DIR/rung1_line_crosswalk.csv" \
    --genelist "$STACK_GENELIST" \
    --gene-metadata "$RUNG1_CACHE/metadata/gene_metadata.parquet" \
    --cache "$RUNG1_CACHE" \
    --block "$SLURM_ARRAY_TASK_ID" \
    --n-blocks "$N_DMSO_BLOCKS"
