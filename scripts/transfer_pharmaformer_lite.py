"""PharmaFormer-lite: drug-structure-aware bilinear transfer, GDSC2 -> Soragni.

Our identity-based per-drug transfer could only touch the 16 drugs shared between
the cohorts and treated each as an opaque label. PharmaFormer's key ingredient is
*drug structure*: encoding the molecule lets the model share signal across drugs
and predict drugs unseen in training. CoderData ships 1024-bit Morgan
fingerprints, so we add them in a simple bilinear (discriminative) model:

    AUC(s, d) = ridge( [ z_s , g_d , z_s (x) g_d ] )

z_s = PCA of log1p expression, g_d = PCA of the drug's Morgan fingerprint, both
fit on GDSC2 and frozen; the outer product z_s (x) g_d is the drug-specific
interaction, now mediated by chemistry rather than drug identity. Trained on
GDSC2 (all drugs), predicted on Soragni -- scored on all 26 drugs and on the 16
shared (for direct comparison with the identity transfer, interaction 0.090).

  uv run python scripts/transfer_pharmaformer_lite.py
  uv run python scripts/transfer_pharmaformer_lite.py --pan-cancer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fmharness.controls import permute_within_drug
from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, interaction_rho, within_drug_rho

SEED = 0
GDSC_SARCOMA = [
    "Alveolar Rhabdomyosarcoma",
    "Chondrosarcoma",
    "Ewing's Sarcoma",
    "Osteosarcoma",
    "Other Sarcomas",
    "Rhabdomyosarcoma",
]
_ALPHAS = tuple(float(a) for a in np.logspace(-1.0, 6.0, 8))


def load_fingerprints(repo: Path) -> pd.DataFrame:
    """PubChem CID -> 1024-bit Morgan fingerprint, unioned across both tranches.

    Keyed by CID so it joins the CID-keyed designs (build_sample_design with
    drug_key='pubchem_cid'). CoderData's drug_descriptors are keyed by
    improve_drug_id, which the matching drugs table maps to a PubChem CID.
    """
    frames = []
    for ds in ("gdscv2", "sarcoma"):
        d = pd.read_csv(repo / f"data/raw/coderdata/{ds}_drug_descriptors.tsv.gz", sep="\t")
        d = d[d["structural_descriptor"] == "morgan fingerprint"]
        drugs = pd.read_csv(repo / f"data/raw/coderdata/{ds}_drugs.tsv.gz", sep="\t")
        id2cid = {
            str(i): str(int(c))
            for i, c in zip(drugs["improve_drug_id"], drugs["pubchem_id"], strict=False)
            if pd.notna(c)
        }
        frames.append(d.assign(cid=d["improve_drug_id"].astype(str).map(id2cid)))
    mf = pd.concat(frames).dropna(subset=["cid"]).drop_duplicates("cid")
    bits = np.array([[int(c) for c in str(v)] for v in mf["descriptor_value"]], dtype=np.float64)
    return pd.DataFrame(bits, index=pd.Index(mf["cid"].astype(str)))


def _features(design: pd.DataFrame, zdf: pd.DataFrame, gdf: pd.DataFrame) -> np.ndarray:
    z = zdf.loc[design["patient"]].to_numpy()
    g = gdf.loc[design["drug"].astype(str)].to_numpy()
    inter = np.einsum("ij,ik->ijk", z, g).reshape(len(z), -1)
    return np.hstack([z, g, inter])


def _score(preds: pd.DataFrame, n_perm: int) -> tuple[float, float, float, float, int]:
    t, p = preds["y_true"].to_numpy(float), preds["y_pred"].to_numpy(float)
    gs = float(np.asarray(spearmanr(t, p))[0])
    wd = within_drug_rho(preds, "y_pred")
    it = interaction_rho(preds, "y_pred")
    null = np.empty(n_perm)
    for b in range(n_perm):
        rng = np.random.default_rng(SEED + 1 + b)
        q = preds.copy()
        q["y_true"] = permute_within_drug(preds["drug"], preds["y_true"], rng)
        null[b] = interaction_rho(q, "y_pred")
    return gs, wd, it, float(np.mean(null >= it)), int(preds["drug"].nunique())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kz", type=int, default=20, help="expression PCs")
    ap.add_argument("--kg", type=int, default=20, help="fingerprint PCs")
    ap.add_argument("--pan-cancer", action="store_true", help="train on all GDSC2")
    ap.add_argument("--n-permutations", type=int, default=1000)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    # Join GDSC2 <-> Soragni (and the fingerprint table) on PubChem CID.
    xs, ds = build_sample_design(
        load_tranche("sarcoma", repo), "organoid", "viability", drug_key="pubchem_cid"
    )
    ctf = None if args.pan_cancer else GDSC_SARCOMA
    gbun = load_tranche("gdscv2", repo, cancer_type_filter=ctf)
    xg, dg = build_sample_design(gbun, "all", "auc", drug_key="pubchem_cid")
    fp = load_fingerprints(repo)

    # keep rows whose drug has a fingerprint
    dg = dg[dg["drug"].astype(str).isin(fp.index)].copy()
    ds = ds[ds["drug"].astype(str).isin(fp.index)].copy()

    # z: expression PCA fit on GDSC2 (shared genes), applied to both, frozen
    genes = sorted(set(xs.columns) & set(xg.columns))
    zpipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(args.kz, random_state=SEED))])
    zg = pd.DataFrame(zpipe.fit_transform(np.log1p(xg[genes])), index=xg.index)
    zs = pd.DataFrame(zpipe.transform(np.log1p(xs[genes])), index=xs.index)

    # g: fingerprint PCA fit on GDSC2's drugs, applied to all, frozen
    gdrugs = sorted(set(dg["drug"].astype(str)))
    gpca = PCA(min(args.kg, len(gdrugs) - 1), random_state=SEED)
    gpca.fit(fp.loc[gdrugs])
    gdf = pd.DataFrame(gpca.transform(fp), index=fp.index)

    # fit bilinear ridge on GDSC2, predict Soragni
    fpipe = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=np.asarray(_ALPHAS)))])
    fpipe.fit(_features(dg, zg, gdf), dg["y"].to_numpy(float))
    pred = fpipe.predict(_features(ds, zs, gdf))
    preds = pd.DataFrame(
        {
            "patient": list(ds["patient"]),
            "drug": list(ds["drug"]),
            "y_true": ds["y"].to_numpy(float),
            "y_pred": pred,
        }
    )

    # in-distribution sanity: does the model capture interaction on held-out GDSC2 lines?
    rng = np.random.default_rng(SEED)
    lines = np.array(sorted(dg["patient"].unique()))
    hold = set(rng.choice(lines, size=max(2, len(lines) // 5), replace=False))
    tr, te = dg[~dg["patient"].isin(hold)], dg[dg["patient"].isin(hold)]
    chk = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=np.asarray(_ALPHAS)))])
    chk.fit(_features(tr, zg, gdf), tr["y"].to_numpy(float))
    te_pred = chk.predict(_features(te, zg, gdf))
    idp = pd.DataFrame(
        {
            "patient": list(te["patient"]),
            "drug": list(te["drug"]),
            "y_true": te["y"].to_numpy(float),
            "y_pred": te_pred,
        }
    )

    shared = sorted(set(ds["drug"].astype(str)) & set(dg["drug"].astype(str)))
    shared_sub = preds[preds["drug"].astype(str).isin(shared)]
    train_kind = "pan-cancer" if args.pan_cancer else "sarcoma"
    print(
        f"train: GDSC2 {train_kind} {dg['patient'].nunique()} lines x "
        f"{dg['drug'].nunique()} drugs ({len(dg)} pairs) | test: Soragni {len(ds)} rows"
    )
    print(
        f"in-dist GDSC2 held-out lines: interaction {interaction_rho(idp, 'y_pred'):+.3f} "
        f"(n={len(te)})"
    )
    print(f"\n{'drug set':16s}{'n_drugs':>8}{'global_sp':>10}{'within':>9}{'interact':>10}{'p':>8}")
    for label, sub in (("all soragni", preds), ("16 shared", shared_sub)):
        gs, wd, it, pv, nd = _score(sub, args.n_permutations)
        print(f"{label:16s}{nd:>8d}{gs:>+10.3f}{wd:>+9.3f}{it:>+10.3f}{pv:>8.3f}")


if __name__ == "__main__":
    main()
