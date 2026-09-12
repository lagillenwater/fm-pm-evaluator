#!/bin/bash
#SBATCH --job-name=rung1-simple-gen
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1                  # 0 stack_cytokine, 1 stack_sciplex
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Rung 1, generation stage: Stack in in-context generation mode, one checkpoint per array
# task, one 512-cell set per (drug, held-out line). Builds the tensors first if the cache is
# empty. The score stage is rung1_simple_score.sbatch, submitted with --dependency=afterok on
# this array. Results land on scratch; nothing is committed by these jobs.
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source scripts/alpine/rung1_env.sh

CACHE="${RUNG1_SIMPLE_CACHE:-/scratch/alpine/$USER/rung1_simple/cache_v2}"
OUT_DIR="${RUNG1_SIMPLE_OUT:-/scratch/alpine/$USER/rung1_simple/results_v3}"
MODELS=(stack_cytokine stack_sciplex)
GEN_MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID model=$GEN_MODEL cache=$CACHE out=$OUT_DIR"

python scripts/rung1_simple_tables.py --stage build \
    --rung0-cache "/scratch/alpine/$USER/rung0_cache_v3b" --tahoe-dir "$TAHOE_DE_DIR" \
    --genelist "$STACK_GENELIST" --cache "$CACHE" --out "$OUT_DIR" --duckdb-memory 40GB --duckdb-threads 8

python scripts/rung1_simple_tables.py --stage generate --gen-model "$GEN_MODEL" --gen-batch 2 \
    --rung0-cache "/scratch/alpine/$USER/rung0_cache_v3b" --tahoe-dir "$TAHOE_DE_DIR" \
    --genelist "$STACK_GENELIST" --cache "$CACHE" --out "$OUT_DIR"
