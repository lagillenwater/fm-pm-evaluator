#!/bin/bash
#SBATCH --job-name=rung1-answers
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=75G                    # 20 cores x 3,840 MB = 76,800 MB = 75G exactly. DuckDB
                                     # below runs with a 20GB memory-limit; PROCESS §2 measured
                                     # this engine peaking 35-40GB above its own limit on this
                                     # 89GB table, so 20GB + that overhead is ~55-60GB, inside
                                     # 75G. The sibling crosswalk call (rung1_grid.sbatch) OOM'd
                                     # in job 32422188 at --mem=30G with no --memory-limit set
                                     # (MaxRSS 31.5GB against DuckDB's unbounded 20GB default);
                                     # this job already passes --memory-limit/--threads below,
                                     # but the same overhead rule sizes its --mem too.
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
