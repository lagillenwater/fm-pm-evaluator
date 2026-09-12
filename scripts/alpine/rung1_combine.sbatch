#!/bin/bash
#SBATCH --job-name=rung1-combine
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G                    # 4 cores x 3,840 MB = 15,360 MB = 15G exactly. Task 10b
                                     # measured this step at 8 GB, which needs 3 cores; 4 is the
                                     # first whole-core request landing on an exact GiB figure.
                                     # The controls dominate, not the gather: the fit control
                                     # holds a 50 x 107 x 300 float64 grid several times over
                                     # plus models.MAX_BYTES (2 GiB) of ridge working arrays.
#SBATCH --time=03:00:00              # task 10b measured about an hour, CPU only (its controls
                                     # took ~20 s on a laptop); the risk is scratch I/O over the
                                     # 165 cached files, not compute
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 1, stage 7: one job that gathers the 157 rounds and the 8 redraw blocks into the run's
# result tables, the five control tables, the five figures, the leakage records and the
# parameter sidecar. Submitted with --dependency=afterok:<redraw array id> -- the WHOLE array:
# an interval, a p-value or an MDE taken over some of the blocks is a different statistic from
# the one the design declares, so the combine refuses a partial set. It also reads the build
# stages' tables from $OUT_DIR (the ceiling, the cells, the identity matches and their grids,
# the weights check) and refuses, naming them, if any is missing. A combine whose outputs are
# all done skips without gathering.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_JOB_ID:-} cache=$RUNG1_CACHE out_dir=$OUT_DIR"

python scripts/heldout_combine.py \
    --grid "$OUT_DIR/rung1_grid.json" \
    --cache "$RUNG1_CACHE" \
    --drug-metadata "$RUNG1_CACHE/metadata/drug_metadata.parquet" \
    --out-dir "$OUT_DIR"
