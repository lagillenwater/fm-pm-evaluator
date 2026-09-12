#!/bin/bash
#SBATCH --job-name=rung1-fit
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G                    # 4 cores x 3,840 MB = 15,360 MB = 15G exactly (Slurm's G
                                     # is GiB). A round holds the answers -- delta float32
                                     # ~0.96 GB, delta0 another ~0.96 GB, tested ~0.24 GB, about
                                     # 2.2 GB in all (task 10a) -- and fits inside
                                     # models.MAX_BYTES (2 GiB) of working blocks, so its floor
                                     # is ~4.3 GiB. Task 11B's brief sizes a task at 8 GB or
                                     # more; that needs 3 cores (11,520 MB), and 4 is the first
                                     # whole-core request landing on an exact GiB figure.
#SBATCH --time=02:00:00              # 14 (leave one line out) or 13 (leave one drug out) models
                                     # per round; task 10a measured the ridge fits at 6-23 s
                                     # each at full size, so the compute is minutes. The rest is
                                     # slack for reading answers.npz off scratch, which rung 0
                                     # saw run 3x slower on 2026-09-10.
#SBATCH --array=0-156                # 157 rounds: 50 lines (0-49), then 107 drugs (50-156)
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 1, stage 5: one job-array task per round -- one line, or one drug, hidden -- fitting and
# scoring every model declared for that scheme. --array MUST cover both schemes' rounds; the
# redraw stage refuses to run until all 157 rounds have valid completion records, so it depends
# on this WHOLE array and never on one index. A round already done skips, so a failed index is
# resubmitted alone with --array=<index>.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

# The design's grid (design.md section 2), the same two numbers heldout_fit.py declares as
# FULL_GRID_LINES and FULL_GRID_DRUGS; tests/test_rung1_jobs.py cross-checks them against it.
N_LOLO_ROUNDS=50                     # indexes 0-49 hide a line
N_LODO_ROUNDS=107                    # indexes 50-156 hide a drug
round_for_index() {
    # One array index -> "<scheme> <round>", the two arguments heldout_fit.py is called with.
    # An index outside the grid stops the task: falling through to round 0 would rewrite a
    # finished round's outputs under another index's name.
    local index="$1"
    if (( index < 0 || index >= N_LOLO_ROUNDS + N_LODO_ROUNDS )); then
        echo "array index $index is outside 0..$(( N_LOLO_ROUNDS + N_LODO_ROUNDS - 1 ))" >&2
        return 1
    fi
    if (( index < N_LOLO_ROUNDS )); then
        printf 'lolo %s\n' "$index"
    else
        printf 'lodo %s\n' "$(( index - N_LOLO_ROUNDS ))"
    fi
}

DISPATCH="$(round_for_index "${SLURM_ARRAY_TASK_ID:?rung1_fit.sbatch needs SLURM_ARRAY_TASK_ID}")"
SCHEME="${DISPATCH%% *}"
ROUND="${DISPATCH##* }"

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-} scheme=$SCHEME round=$ROUND cache=$RUNG1_CACHE"

python scripts/heldout_fit.py \
    --scheme "$SCHEME" \
    --round "$ROUND" \
    --grid "$OUT_DIR/rung1_grid.json" \
    --cache "$RUNG1_CACHE"
