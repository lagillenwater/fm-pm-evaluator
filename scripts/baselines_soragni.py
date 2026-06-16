"""Simple reference baselines for Soragni (organoid x drug) AUC.

Scored exactly like Path B -- global rho (drug-level: are cytotoxic drugs called
cytotoxic) and interaction rho (organoid-specific: right drug for the right tumor,
the part a drug-average cannot produce), with the within-drug label-permutation
null -- so the rows sit directly under the Stack-generation result.

  drug-mean      leave-one-organoid-out per-drug mean AUC (no expression). Pure
                 drug-level floor; interaction ~0 by construction.
  l1000:<sig>    the REAL measured L1000 average drug delta (treated - DMSO) read
                 through a death signature. Organoid-independent, so it shows the
                 drug-level signal Stack inherits for free; interaction ~0. This is
                 the floor Stack must BEAT (by adding organoid-specificity).
  pca / nmf      GDSC2-trained, FROZEN bilinear predictor with PCA/NMF of
                 pre-treatment expression as the organoid representation (per-drug
                 head, so it can carry interaction). Never sees Soragni AUC.

  uv run python scripts/baselines_soragni.py \\
      --l1000-context l1000_context.h5ad --signatures hallmark
"""

from __future__ import annotations

import argparse
import urllib.request
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.controls import permute_within_drug
from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, global_spearman, interaction_rho
from fmharness.probe import SimpleProbe
from fmharness.signatures import load_hallmark, sensitivity_scores

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


def score(preds: pd.DataFrame, n_perm: int) -> tuple[float, float, float]:
    """global rho, interaction rho, within-drug label-permutation p (vs interaction)."""
    gl = global_spearman(preds)
    it = interaction_rho(preds, "y_pred")
    null = np.array(
        [
            interaction_rho(
                preds.assign(
                    y_true=permute_within_drug(
                        preds["drug"], preds["y_true"], np.random.default_rng(SEED + 1 + b)
                    )
                ),
                "y_pred",
            )
            for b in range(n_perm)
        ]
    )
    return gl, it, float(np.mean(null >= it))


def transfer_predict(factory, fg, dg, fs, ds) -> pd.DataFrame:
    """Fit a probe on GDSC2, predict the frozen Soragni cohort."""
    probe = factory()
    probe.fit(
        fg.loc[dg["patient"]].to_numpy(),
        list(dg["drug"]),
        dg["y"].to_numpy(),
        groups=list(dg["patient"]),
    )
    base, resid = probe.predict_parts(fs.loc[ds["patient"]].to_numpy(), list(ds["drug"]))
    return pd.DataFrame(
        {
            "patient": list(ds["patient"]),
            "drug": list(ds["drug"]),
            "y_true": ds["y"].to_numpy(dtype=np.float64),
            "y_pred": base + resid,
        }
    )


def drug_mean(ds: pd.DataFrame) -> pd.DataFrame:
    """Per-drug mean AUC -- organoid-independent, so interaction is ~0 by construction.
    The drug-level floor: global rho is how much drug identity alone explains."""
    pred = ds.groupby("drug")["y"].transform("mean")
    return pd.DataFrame(
        {
            "patient": ds["patient"],
            "drug": ds["drug"],
            "y_true": ds["y"].to_numpy(dtype=np.float64),
            "y_pred": pred.to_numpy(),
        }
    )


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


def direct_l1000(
    ctx_path: Path,
    ds: pd.DataFrame,
    sigs,
    repo: Path,
) -> list[tuple[str, pd.DataFrame]]:
    """Real L1000 average drug delta read through each signature, broadcast to organoids."""
    ctx = ad.read_h5ad(ctx_path)
    x = _dense(ctx.X)
    genes = [str(g) for g in ctx.var_names]
    is_ctrl = ctx.obs["is_control"].to_numpy().astype(bool)
    pert = ctx.obs["pert_id"].astype(str).to_numpy()
    dmso = x[is_ctrl].mean(axis=0)
    p2d = pertid_to_drug(repo, set(ds["drug"].astype(str)))
    rows, names = [], []
    for pid in np.unique(pert[~is_ctrl]):
        if pid in p2d:
            rows.append(x[(pert == pid) & ~is_ctrl].mean(axis=0) - dmso)
            names.append(p2d[pid])
    delta = pd.DataFrame(rows, columns=pd.Index(genes), index=pd.Index(names))
    sens = sensitivity_scores(delta, sigs)  # drugs x signatures
    out: list[tuple[str, pd.DataFrame]] = []
    for sig in sens.columns:
        m = ds.assign(y_pred=-ds["drug"].astype(str).map(sens[sig]).to_numpy()).dropna(
            subset=["y_pred"]
        )
        out.append(
            (
                f"l1000:{sig}",
                pd.DataFrame(
                    {
                        "patient": m["patient"],
                        "drug": m["drug"],
                        "y_true": m["y"].to_numpy(dtype=np.float64),
                        "y_pred": m["y_pred"].to_numpy(),
                    }
                ),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-context", default=None, help="context .h5ad for direct-L1000")
    ap.add_argument("--signatures", choices=["curated", "hallmark"], default="hallmark")
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--std-floor", type=float, default=0.5)
    ap.add_argument("--n-permutations", type=int, default=1000)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    sigs = (
        load_hallmark(repo / "data/static/hallmark_signatures.gmt")
        if args.signatures == "hallmark"
        else None
    )
    xs, ds = build_sample_design(load_tranche("sarcoma", repo), "organoid", "viability")
    xg, dg = build_sample_design(load_tranche("gdscv2", repo), "all", "auc")
    gdsc_drugs = set(dg["drug"].astype(str))

    # direct-L1000 defines Path B's drug set (the L1000-matched drugs) when a context is given
    l1000_rows = (
        direct_l1000(Path(args.l1000_context), ds, sigs, repo) if args.l1000_context else []
    )
    ref = (
        set(l1000_rows[0][1]["drug"].astype(str))
        if l1000_rows
        else set(ds["drug"].astype(str)) & gdsc_drugs
    )

    rows: list[tuple[str, pd.DataFrame]] = [
        ("drug-mean", drug_mean(ds[ds["drug"].astype(str).isin(ref)].copy())),
        *l1000_rows,
    ]
    # PCA/NMF transfer is limited to ref drugs GDSC2 also screened
    shared = sorted(ref & gdsc_drugs)
    ds_t = ds[ds["drug"].astype(str).isin(shared)].copy()
    dg_t = dg[dg["drug"].astype(str).isin(shared)].copy()
    genes = sorted(set(xs.columns) & set(xg.columns))
    fg, fs = np.log1p(xg[genes]), np.log1p(xs[genes])
    print(f"ref drugs {len(ref)} | transfer (ref & GDSC2) drugs {len(shared)}")
    for rep, reducer in (("pca", "pca"), ("nmf", "nmf")):
        factory = partial(
            SimpleProbe,
            n_components=args.n_components,
            std_floor=args.std_floor,
            reducer=reducer,
            per_drug=True,
        )
        rows.append((rep, transfer_predict(factory, fg, dg_t, fs, ds_t)))

    print(f"\n=== Soragni baselines ({args.signatures} signature) ===")
    print(f"{'baseline':22s}{'global':>9}{'interact':>10}{'p_label':>9}{'n':>6}")
    for name, preds in rows:
        gl, it, pv = score(preds, args.n_permutations)
        print(f"{name:22s}{gl:>+9.3f}{it:>+10.3f}{pv:>9.3f}{len(preds):>6d}")


if __name__ == "__main__":
    main()
