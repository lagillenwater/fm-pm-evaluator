# Sourced by every rung 1 data-stage job script: the environment those stages share.
#
# Activate the `stack` env by PATH instead of the module system, as rung 0 does (`module load
# anaconda` fails inside batch jobs for reasons NOT established; this pattern is the one that is
# PROVEN to work). `conda activate` is deliberately not used: this install ships no
# etc/profile.d/conda.sh.
export PATH="/projects/$USER/software/anaconda/envs/stack/bin:$PATH"
export CONDA_PREFIX="/projects/$USER/software/anaconda/envs/stack"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1   # else stdout block-buffers to the log file and a multi-hour job
                           # shows 0 bytes until it exits (rung 0, job 31655278, 2026-08-25).
export HF_HUB_CACHE="/scratch/alpine/$USER/hf"
# do NOT set HF_HOME to scratch -- that hides the token `hf auth login` saved under the HF cache:
export HF_TOKEN="${HF_TOKEN:-$(cat "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/token" 2>/dev/null || true)}"

# Pinned to the pyproject floors with upper bounds so a fallback install can't drift the
# persistent stack env to an untested major version. anndata, torch and stack are already in
# the env (rung 1's GPU stage needs them); everything else rung 1 adds is checked here.
python -c "import duckdb" 2>/dev/null || pip install -q "duckdb>=1.0,<2"
python -c "import matplotlib" 2>/dev/null || pip install -q "matplotlib>=3.8,<4"
python -c "import rdkit" 2>/dev/null || pip install -q "rdkit>=2024.3,<2026"
python -c "import pydantic" 2>/dev/null || pip install -q "pydantic>=2.7,<3"
python -c "import sklearn" 2>/dev/null || pip install -q "scikit-learn>=1.4,<2"
python -c "import scipy" 2>/dev/null || pip install -q "scipy>=1.11,<2"

# The data stages share one cache. It lives on scratch: it is a speed optimisation, not the
# reproduction chain -- the pinned tranches plus the committed code are that -- and scratch is
# purged.
export RUNG1_CACHE="${RUNG1_CACHE:-/scratch/alpine/$USER/rung1_cache}"
export OUT_DIR="${OUT_DIR:-docs/tasks/rung1-held-out-prediction}"

# Rung 0's pseudobulk DE table, scanned again for rung 1's filters (5 uM, 107 drugs, 50 lines).
export TAHOE_DE_DIR="${TAHOE_DE_DIR:-/scratch/alpine/$USER/tahoe_pseudobulk_de}"

# Stack's gene panel and the three checkpoints task 6 embeds with.
export STACK_GENELIST="${STACK_GENELIST:-stack-large/basecount_1000per_15000max.pkl}"
export CKPT_BASE="${CKPT_BASE:-stack-large/bc_large.ckpt}"
export CKPT_CYTOKINE="${CKPT_CYTOKINE:-stack-aligned/bc_large_aligned.ckpt}"
export CKPT_DRUG="${CKPT_DRUG:-/scratch/alpine/$USER/sciplex_finetune/finetuned-epoch=5-val_loss=6.1078.ckpt}"

# The array sizes in one place: each array's --array range and this script's constants must
# agree, so both the sbatch files and the chain submitter read them from here (PROCESS §2).
export N_ANSWER_PARTS="${N_ANSWER_PARTS:-8}"
export N_DMSO_BLOCKS="${N_DMSO_BLOCKS:-64}"
export DMSO_CONCURRENCY="${DMSO_CONCURRENCY:-4}"

mkdir -p logs "$RUNG1_CACHE"
