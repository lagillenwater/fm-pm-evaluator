#!/bin/bash
#SBATCH --job-name=rung1-grid
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=75G                    # 20 cores x 3,840 MB = 76,800 MB = 75G exactly. The
                                     # crosswalk scan below runs heldout_answers.py's DuckDB
                                     # engine at a 20GB memory-limit; PROCESS §2 measured this
                                     # engine peaking 35-40GB above its own limit on this 89GB
                                     # table, so 20GB + that overhead is ~55-60GB, inside 75G.
                                     # Job 32422188 OOM'd at --mem=30G (MaxRSS 31.5GB) because
                                     # this call passed no --memory-limit/--threads and ran on
                                     # DuckDB's unbounded defaults (20GB limit, 4 threads) with
                                     # no headroom for the overhead.
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
    --out-dir "$OUT_DIR" \
    --memory-limit 20GB \
    --threads 4
