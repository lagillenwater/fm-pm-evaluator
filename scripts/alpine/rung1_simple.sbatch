#!/bin/bash
#SBATCH --job-name=rung1-simple
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Rung 0 noise per drug / per line and rung 1 held-out prediction, three tables, one job:
# duckdb over the rung 0 frame cache and every third raw shard, three Stack embeddings of synthetic cells drawn from each of the 49
# lines' baseline, then the fold loop. Results land on scratch; pull the CSVs from
# $OUT_DIR, nothing is committed by this job.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="${RUNG1_SIMPLE_CACHE:-/scratch/alpine/$USER/rung1_simple/cache_v2}"
OUT_DIR="${RUNG1_SIMPLE_OUT:-/scratch/alpine/$USER/rung1_simple/results_v2}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID cache=$CACHE out=$OUT_DIR"

python scripts/rung1_simple_tables.py \
    --stage "${STAGE:-all}" \
    --rung0-cache "/scratch/alpine/$USER/rung0_cache_v3b" \
    --tahoe-dir "$TAHOE_DE_DIR" \
    --genelist "$STACK_GENELIST" \
    --cache "$CACHE" \
    --out "$OUT_DIR" \
    --duckdb-memory 40GB \
    --duckdb-threads 8
