#!/bin/bash
#SBATCH --job-name=rung0-assign
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=84G                    # peaked at 83G with the engine at 60G (job 32365513's
                                     # predecessor 32341610); peak runs ~35G above the engine
                                     # limit, so 40G below keeps it inside. Memory is billed
                                     # per core at 3,840 MB: 84G is a 22-core job, 120G a 32
#SBATCH --time=00:40:00              # it ran 13 min; a short limit backfills into small gaps
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 0, stage 1 of 3: the split assignment. One scan over the key columns alone (line, drug,
# dose, plate, and baseMean for the untestable share) -- about a quarter of an hour -- and the
# plate-to-half assignment becomes an explicit table every slice joins against: plates sorted by
# id within each (line, drug, dose) triple, assigned alternately. It also writes the pool
# description (plates per triple, per half) into the task folder.
#
# Chain: this job -> rung0_slice.sbatch (array, afterok) -> rung0_combine.sbatch (afterok).
# scripts/alpine/submit_rung0_chain.sh submits all three with the dependencies set.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung0_env.sh

python scripts/delta_reproducibility.py \
    --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
    --stage assign \
    --cache-dir "$RUNG0_CACHE" \
    --duckdb-memory 40GB \
    --duckdb-threads 4 \
    --out-dir "$OUT_DIR"
