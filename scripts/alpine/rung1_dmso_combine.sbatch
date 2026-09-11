#!/bin/bash
#SBATCH --job-name=rung1-dmso-combine
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G                    # 8 cores x 3,840 MB; sums the blocks' counts, selects
                                     # cells, and writes at most 1,000 cells x 50 lines
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 1, stage 2b: refuses (non-zero exit) unless every one of the DMSO-cells array's blocks
# is done, then assembles the plate-balanced, capped selection into per-line AnnData files and
# pseudobulk expression tables, and registers the tranche under data/tranches/ in this checkout
# (a tranche is ingested once, then immutable -- a rerun after registration fails loudly rather
# than silently re-registering). Submitted with --dependency=afterok:<dmso-cells array id>.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_JOB_ID:-} n_blocks=$N_DMSO_BLOCKS cache=$RUNG1_CACHE out_dir=$OUT_DIR"

python scripts/heldout_dmso_cells.py \
    --combine \
    --grid "$OUT_DIR/rung1_grid.json" \
    --crosswalk "$OUT_DIR/rung1_line_crosswalk.csv" \
    --genelist "$STACK_GENELIST" \
    --gene-metadata "$RUNG1_CACHE/metadata/gene_metadata.parquet" \
    --cache "$RUNG1_CACHE" \
    --out-dir "$OUT_DIR" \
    --n-blocks "$N_DMSO_BLOCKS" \
    --tranche-dir data/tranches
