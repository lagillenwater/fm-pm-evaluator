"""Simple predictors of Soragni TREATED expression, judged exactly like Stack.

Each method predicts the treated transcriptome for every (organoid, drug) from the
SAME inputs Stack used -- the Soragni baseline + the L1000 drug context (NOT GDSC2) --
then goes through the identical bridge: delta = predicted_treated - baseline, read by
the death signature, scored against the real Soragni AUC with the random-gene-set
control (score_signatures). So Stack's generated expression and these simple
predictions are judged the same way, on the same organoid x drug pairs.

  control : predicted treated = baseline (delta 0). Does Stack beat 'no effect'?
  mean    : baseline + the average L1000 drug delta. Organoid-independent -> drug-level
            floor, interaction ~0 by construction. The measured-perturbation floor.
  pca/nmf : a low-rank LINEAR control->treated map fit on the drug's L1000 cell lines,
            applied to each organoid's baseline -> organoid-SPECIFIC predicted treated.
            The 'does the foundation model beat a linear predictor' test. L1000 and
            Soragni are bridged by per-gene standardization into a shared space; if that
            naive bridge fails, that is itself the platform gap Stack is meant to close.

  uv run python scripts/predict_expression_baselines.py \\
      --l1000-context l1000_context_rich.h5ad \\
      --baseline data/reference/stack_input_soragni.h5ad --signatures hallmark
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import Ridge

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design
from fmharness.signatures import load_hallmark, score_signatures

SEED = 0
PERT_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_pert_info.txt.gz"
)


def _dense(x: object) -> np.ndarray:
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.asarray(arr, dtype=np.float64)


def _ncid(x: object) -> str:
    try:
        return str(int(float(x)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return ""


def pertid_to_drug(repo: Path, drugs: set[str]) -> dict[str, str]:
    """L1000 pert_id -> Soragni improve_drug_id, via PubChem CID / InChIKey prefix."""
    cache = Path("/tmp/l1000_pert_info.txt.gz")
    if not cache.exists():
        urllib.request.urlretrieve(PERT_INFO_URL, cache)
    cp = pd.read_csv(cache, sep="\t", low_memory=False)
    cp = cp[cp["pert_type"] == "trt_cp"]
    cid2p = {_ncid(c): p for c, p in zip(cp["pubchem_cid"], cp["pert_id"], strict=True)}
    ikb2p = {str(k): p for k, p in zip(cp["inchi_key_prefix"], cp["pert_id"], strict=True)}
    dr = pd.read_csv(repo / "data/raw/coderdata/sarcoma_drugs.tsv.gz", sep="\t")
    sor = dr[dr["improve_drug_id"].astype(str).isin(drugs)].drop_duplicates("improve_drug_id")
    out: dict[str, str] = {}
    for _, r in sor.iterrows():
        pid = cid2p.get(_ncid(r["pubchem_id"])) or ikb2p.get(str(r["InChIKey"])[:14])
        if pid:
            out[pid] = str(r["improve_drug_id"])
    return out


def _broadcast(
    per_drug: dict[str, np.ndarray],
    orgs: list[str],
    genes: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a {drug: delta_vector} (organoid-independent) into aligned delta/key frames."""
    rows, key = [], []
    for drug, vec in per_drug.items():
        for org in orgs:
            rows.append(vec)
            key.append((org, drug))
    return (pd.DataFrame(rows, columns=genes), pd.DataFrame(key, columns=["patient", "drug"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-context", required=True, help="L1000 context .h5ad (rich preferred)")
    ap.add_argument("--baseline", default="data/reference/stack_input_soragni.h5ad")
    ap.add_argument("--signatures", choices=["curated", "hallmark"], default="hallmark")
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--min-lines", type=int, default=5, help="min L1000 cell lines for the map")
    ap.add_argument("--n-permutations", type=int, default=1000)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    sigs = (
        load_hallmark(repo / "data/static/hallmark_signatures.gmt")
        if args.signatures == "hallmark"
        else None
    )
    _, design = build_sample_design(load_tranche("sarcoma", repo), "organoid", "viability")

    base = ad.read_h5ad(
        repo / args.baseline if not Path(args.baseline).is_absolute() else Path(args.baseline)
    )
    ctx = ad.read_h5ad(args.l1000_context)
    genes = pd.Index([str(g) for g in base.var_names]).intersection([str(g) for g in ctx.var_names])
    bmap = {str(g): i for i, g in enumerate(base.var_names)}
    cmap = {str(g): i for i, g in enumerate(ctx.var_names)}
    bcols = [bmap[g] for g in genes]
    ccols = [cmap[g] for g in genes]
    b = _dense(base.X)[:, bcols]  # organoids x G
    orgs = [str(o) for o in base.obs_names]
    cx = _dense(ctx.X)[:, ccols]  # wells x G
    cell = ctx.obs["cell_id"].astype(str).to_numpy()
    pert = ctx.obs["pert_id"].astype(str).to_numpy()
    isc = ctx.obs["is_control"].to_numpy().astype(bool)
    p2d = pertid_to_drug(repo, set(design["drug"].astype(str)))
    drugs = sorted({p2d[p] for p in np.unique(pert[~isc]) if p in p2d})
    print(
        f"genes {len(genes)} | mapped drugs {len(drugs)} | organoids {len(orgs)} "
        f"| L1000 wells {cx.shape[0]}"
    )

    # per-cell-line DMSO mean (control) and a global DMSO mean
    lines = list(np.unique(cell[isc]))
    ctrl = {c: cx[isc & (cell == c)].mean(0) for c in lines}
    dmso = cx[isc].mean(0)

    # --- shared standardized space: each platform z-scored by its OWN gene stats, so
    # Soragni counts and L1000 Level-3 land in comparable (SD-from-mean) units ---
    cmat = np.array([ctrl[c] for c in lines])
    mu, sd = cmat.mean(0), cmat.std(0)
    sd[sd == 0] = 1.0
    sb_mu, sb_sd = b.mean(0), b.std(0)
    sb_sd[sb_sd == 0] = 1.0
    zb = (b - sb_mu) / sb_sd  # Soragni baseline in its own z-units

    def conditional(reducer: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        zc = (cmat - mu) / sd
        if reducer == "pca":
            red = PCA(n_components=args.n_components, random_state=SEED).fit(zc)
            tx = red.transform
        else:
            shift = float(zc.min())
            nm = NMF(
                n_components=args.n_components, init="nndsvda", max_iter=1000, random_state=SEED
            ).fit(zc - shift)

            def tx(m: np.ndarray) -> np.ndarray:
                return nm.transform(m - shift)

        zb_r = tx(zb)
        rows, key = [], []
        for pid in np.unique(pert[~isc]):
            if pid not in p2d:
                continue
            have = [c for c in np.unique(cell[(pert == pid) & ~isc]) if c in ctrl]
            if len(have) < args.min_lines:
                continue
            td = np.array([cx[(pert == pid) & ~isc & (cell == c)].mean(0) for c in have])
            cd = np.array([ctrl[c] for c in have])
            zdelta = (td - cd) / sd
            coef = Ridge(alpha=1.0).fit(tx((cd - mu) / sd), zdelta).coef_  # G x k
            delta_s = zb_r @ coef.T  # organoids x G
            rows.extend(delta_s)
            key.extend((org, p2d[pid]) for org in orgs)
        return pd.DataFrame(rows, columns=genes), pd.DataFrame(key, columns=["patient", "drug"])

    # --- assemble each predictor's delta ---
    methods: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    zero = dict.fromkeys(drugs, np.zeros(len(genes)))
    methods["control"] = _broadcast(zero, orgs, genes)
    mean_delta = {
        p2d[p]: cx[(pert == p) & ~isc].mean(0) - dmso for p in np.unique(pert[~isc]) if p in p2d
    }
    methods["mean"] = _broadcast(mean_delta, orgs, genes)
    methods["pca"] = conditional("pca")
    methods["nmf"] = conditional("nmf")

    res = []
    for name, (delta, key) in methods.items():
        r = score_signatures(delta, key, design, signatures=sigs, n_perm=args.n_permutations)
        r.insert(0, "method", name)
        r["n"] = len(key)
        res.append(r)
    print(f"\n=== Soragni expression-prediction baselines ({args.signatures}) ===")
    print(pd.concat(res, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
