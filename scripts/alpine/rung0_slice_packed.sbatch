#!/bin/bash
#SBATCH --job-name=rung0-slice-packed
#SBATCH --partition=amem
#SBATCH --qos=mem-normal            # amem requires mem-normal/mem-long, and those require a
                                    # job to request 256G or more (CURC Alpine hardware page);
                                    # a whole node is taken and the slices packed into it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=990G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 0, stage 2 of 3, packed: the same thirty-two gene slices as rung0_slice.sbatch, but
# eleven at a time inside one high-memory node instead of one per acpu task. Used when the
# acpu queue is deep (2026-09-10: 1,353 pending, 8 amem nodes idle). Each slice is the same
# `--stage slice` process with its own spill directory and 4 engine threads, run in the
# background and waited on; the pack fails if any slice fails, so the combine's dependency
# holds. A slice already cached skips in seconds, so a pack can be resubmitted alone.
#
# Cost per the CURC billing weights: 48 cores x 4.0 on amem against 32 x 22 cores x 1.0 on
# acpu -- comparable per hour, and it starts when acpu would not.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung0_env.sh

PER_PACK="${RUNG0_PER_PACK:-11}"
first=$(( SLURM_ARRAY_TASK_ID * PER_PACK ))
last=$(( first + PER_PACK - 1 ))
(( last >= RUNG0_SLICES )) && last=$(( RUNG0_SLICES - 1 ))
echo "pack $SLURM_ARRAY_TASK_ID: slices $first..$last of $RUNG0_SLICES on $(hostname)"

pids=()
for part in $(seq "$first" "$last"); do
    python scripts/delta_reproducibility.py \
        --local-dir "/scratch/alpine/$USER/tahoe_pseudobulk_de" \
        --stage slice \
        --slices "$RUNG0_SLICES" \
        --slice-index "$part" \
        --cache-dir "$RUNG0_CACHE" \
        --duckdb-memory 40GB \
        --duckdb-threads 4 \
        --noise-sample-rows 2000000 \
        --out-dir "$OUT_DIR" \
        > "logs/rung0-slice-${SLURM_ARRAY_JOB_ID}_${part}.out" 2>&1 &
    pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
    part=$(( first + i ))
    if wait "${pids[$i]}"; then
        echo "slice $part: done"
    else
        echo "slice $part: FAILED (see logs/rung0-slice-${SLURM_ARRAY_JOB_ID}_${part}.out)"
        failed=1
    fi
done
exit $failed
