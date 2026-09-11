#!/bin/bash
#SBATCH --job-name=rung1-grid
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G                    # 8 cores x 3,840 MB; this stage only writes small tables
                                     # and fetches two metadata files, never scans the DE table
#SBATCH --time=02:00:00              # a Hugging Face metadata pull plus a crosswalk scan of the
                                     # DE table's key columns alone
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 1, stage 0 of the data chain: everything the other data stages read as an input rather
# than compute themselves. Smoke-imports every rung 1 module first (Python 3.10 on this env,
# global-constraints.md), fetches the two Tahoe metadata parquets once so no later job array
# hits Hugging Face for them, writes the grid and ceiling table, mirrors the grid into the
# scratch cache (later stages' --grid arguments may read it from either place), and writes the
# line crosswalk the DMSO-cell stage needs.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_JOB_ID:-} out_dir=$OUT_DIR rung1_cache=$RUNG1_CACHE tahoe_de_dir=$TAHOE_DE_DIR"

python scripts/heldout_smoke_import.py

python scripts/heldout_fetch_metadata.py --out-dir "$RUNG1_CACHE/metadata"

python scripts/heldout_grid.py \
    --per-pair results/rung0-assay-reliability/rung0_per_pair_r.csv \
    --dose-strata results/rung0-assay-reliability/rung0_dose_strata.csv \
    --drug-metadata "$RUNG1_CACHE/metadata/drug_metadata.parquet" \
    --out-dir "$OUT_DIR"

cp "$OUT_DIR/rung1_grid.json" "$RUNG1_CACHE/rung1_grid.json"

python scripts/heldout_answers.py \
    --crosswalk \
    --local-dir "$TAHOE_DE_DIR" \
    --grid "$OUT_DIR/rung1_grid.json" \
    --out-dir "$OUT_DIR"
