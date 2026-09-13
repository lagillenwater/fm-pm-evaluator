#!/bin/bash
#SBATCH --job-name=probe-sciplex-drugs
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# What the sci-Plex fine-tune saw: the compounds and lines in its training input.
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"; cd "$REPO"; source scripts/alpine/rung1_env.sh
python - <<'PY'
import anndata as ad, glob, os
for f in sorted(glob.glob(os.path.expandvars("/scratch/alpine/$USER/sciplex/finetune_input/*.h5ad")) + glob.glob(os.path.expandvars("/scratch/alpine/$USER/sciplex/*.h5ad"))):
    a = ad.read_h5ad(f, backed="r")
    print(f"=== {f}: {a.n_obs} cells; obs columns {list(a.obs.columns)[:12]}")
    for c in [c for c in a.obs.columns if c.lower() in ("pert_id", "product_name", "drug", "perturbation", "cell_line", "cell_type", "condition", "dose", "dose_value", "time")]:
        vals = a.obs[c].astype(str)
        print(f"  {c}: {vals.nunique()} unique -> {sorted(vals.unique())[:200]}")
PY
