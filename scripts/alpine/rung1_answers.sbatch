#!/bin/bash
#SBATCH --job-name=rung1-answers
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G                    # 16 cores x 3,840 MB = 61,440 MB = 60G exactly (Slurm's
                                     # G is GiB; 61G would be 62,464 MB, 1,024 MB over budget);
                                     # rung 0's own slice array (a wider scan of the same table)
                                     # needed 84G at 8 cores, so this slice's DuckDB engine
                                     # limit (20GB) plus its ~35-40G overhead fits inside 60G
#SBATCH --time=04:00:00              # rung 0 saw scratch reads run 3x slower than usual; a
                                     # gene-hash slice of 107 drugs x 50 lines is far smaller
                                     # than rung 0's full scan, but the slack carries over
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 1, stage 1a: one job-array task per gene slice of the answer scan (5 uM, the grid's 107
# drugs and 50 lines). --array here MUST match N_ANSWER_PARTS (rung1_env.sh); the combine job
# reads the same count. A task whose slice is already done skips, so a failed index is
# resubmitted alone with --array=<index>.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} part=${SLURM_ARRAY_TASK_ID:-} n_parts=$N_ANSWER_PARTS cache=$RUNG1_CACHE"

python scripts/heldout_answers.py \
    --local-dir "$TAHOE_DE_DIR" \
    --grid "$OUT_DIR/rung1_grid.json" \
    --cache "$RUNG1_CACHE" \
    --part "$SLURM_ARRAY_TASK_ID" \
    --n-parts "$N_ANSWER_PARTS" \
    --memory-limit 20GB \
    --threads 8
