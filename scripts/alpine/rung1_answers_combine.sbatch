#!/bin/bash
#SBATCH --job-name=rung1-answers-combine
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G                    # 16 cores x 3,840 MB = 61,440 MB = 60G exactly (Slurm's
                                     # G is GiB; 61G would be 62,464 MB, 1,024 MB over budget);
                                     # assembles 8 cached gene-slice parquets into the dense
                                     # answer arrays, no table scan
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 1, stage 1b: refuses (non-zero exit) unless every one of the answers array's slices is
# done, then assembles answers.npz and rung1_answer_counts.csv. Submitted with
# --dependency=afterok:<answers array id> so it waits for every slice.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
echo "Resolved: commit=$COMMIT job=${SLURM_JOB_ID:-} n_parts=$N_ANSWER_PARTS cache=$RUNG1_CACHE out_dir=$OUT_DIR"

python scripts/heldout_answers.py \
    --combine \
    --grid "$OUT_DIR/rung1_grid.json" \
    --cache "$RUNG1_CACHE" \
    --out-dir "$OUT_DIR" \
    --n-parts "$N_ANSWER_PARTS"
