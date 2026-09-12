#!/bin/bash
#SBATCH --job-name=rung1-redraws
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7680M                  # 2 cores x 3,840 MB = 7,680 MB exactly, written in MB
                                     # because 7.5 GiB is no whole "G" figure. A block reads the
                                     # 157 rounds' score files -- about 289,000 rows in all
                                     # (task 10b) -- pivots them once per scheme and gene set,
                                     # and redraws 46 (contrast, gene set) pairs 250 times; the
                                     # draws themselves are [250, 50] and [250, 107] count
                                     # matrices, kilobytes each.
#SBATCH --time=02:00:00              # minutes of arithmetic; the slack is for reading 157
                                     # parquet files off scratch
#SBATCH --array=0-7                  # N_BLOCKS = 8 blocks of DRAWS_PER_BLOCK = 250 draws
                                     # (fmharness.heldout.comparisons), 2,000 draws in all
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 1, stage 6: one job-array task per block of redraws. Every comparison and every
# description-versus-stand-in contrast is redrawn on both gene sets, one-way (the hidden unit)
# and two-way (lines and drugs together). Submitted with --dependency=afterok:<fit array id> --
# the WHOLE array, not one index: heldout_redraws.py refuses to run until all 157 rounds have
# valid completion records, since a comparison taken over some of the rounds is silently a
# different comparison. A block already done skips, so a failed index is resubmitted alone with
# --array=<index>.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} block=${SLURM_ARRAY_TASK_ID:-} cache=$RUNG1_CACHE"

python scripts/heldout_redraws.py \
    --block "$SLURM_ARRAY_TASK_ID" \
    --grid "$OUT_DIR/rung1_grid.json" \
    --cache "$RUNG1_CACHE"
