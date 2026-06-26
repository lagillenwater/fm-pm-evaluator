"""Shared L1000 builders, so the validator, the generated-bridge, and the
viability-adapter runner use one code path (no drift):

- ``build_l1000_gdsc_pairs`` -- real L1000 treated-minus-DMSO deltas paired with
  GDSC2 AUC on shared (cell line, drug) pairs. The validation / supervised-training
  cohort. Reads the Level-3 ``.gctx`` in column chunks (bounded memory); ``cmapPy``
  is imported lazily, so importing this module never requires it (Alpine only).
- ``build_generated_deltas`` -- Stack-generated treated profiles minus the organoid
  baseline, per (organoid, drug). The target cohort. AnnData only.
- ``build_additive_deltas`` -- the non-Stack baseline delta source: each drug's mean
  real L1000 treated-minus-DMSO delta, applied to every organoid (organoid-independent).
  The generation analogue of the drug-mean baseline, so Stack's organoid-specific
  generated delta is compared against "the drug does the same thing everywhere."
- ``build_learned_deltas`` -- PCA/NMF organoid-specific delta predictors: a linear
  baseline -> delta-residual map learned on real L1000, applied to each organoid's
  baseline. The generation analogue of the expression baselines, between the additive
  floor and Stack.

All return a delta frame (rows = samples, columns = gene symbols) plus a key frame
(``patient``, ``drug``) aligned row-for-row, ready for ``score_signatures`` or the
viability adapters -- so any delta source flows through the same readout adapters and
metrics, and the comparison is delta-source vs delta-source on equal footing.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design

PERT_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_pert_info.txt.gz"
)


def dense(x: object) -> np.ndarray:
    """AnnData.X may be sparse; return a dense 2-D float array."""
    to_array = getattr(x, "toarray", None)
    arr = to_array() if callable(to_array) else np.asarray(x)
    return np.asarray(arr, dtype=np.float64)


def logcpm(df: pd.DataFrame) -> pd.DataFrame:
    """Library-size normalize (per 10k) and log1p, so a treated-minus-baseline
    difference is a log fold-change rather than a raw-count difference dominated by
    per-sample sequencing depth (which would inflate the random-gene-set baseline)."""
    lib = df.sum(axis=1).to_numpy(dtype=np.float64)
    lib[lib == 0] = 1.0
    return pd.DataFrame(
        np.log1p(df.to_numpy(dtype=np.float64) / lib[:, None] * 1e4),
        index=df.index,
        columns=df.columns,
    )


def _ncid(x: object) -> str:
    try:
        return str(int(float(x)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return ""


def _norm(s: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def drug_pert_maps(
    drugs: pd.DataFrame,
    pert_info: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """``(drug2pert, pert2drug)`` mapping **PubChem CID** (string) to L1000
    ``trt_cp`` pert_ids by PubChem CID or InChIKey prefix. CID is the canonical
    cross-dataset drug key the designs use (build_sample_design
    drug_key='pubchem_cid'); drugs without a CID are skipped. ``drugs`` needs
    columns ``pubchem_id``, ``InChIKey``."""
    cp = pert_info[pert_info["pert_type"] == "trt_cp"]
    cid2p = {_ncid(c): p for c, p in zip(cp["pubchem_cid"], cp["pert_id"], strict=True)}
    ikb2p = {str(k): p for k, p in zip(cp["inchi_key_prefix"], cp["pert_id"], strict=True)}
    drug2pert: dict[str, str] = {}
    pert2drug: dict[str, str] = {}
    for _, r in drugs.drop_duplicates(subset=["pubchem_id", "InChIKey"]).iterrows():
        try:
            cid = str(int(r["pubchem_id"]))  # skips None / NaN (no canonical CID)
        except (TypeError, ValueError):
            continue
        pid = cid2p.get(_ncid(r["pubchem_id"])) or ikb2p.get(str(r["InChIKey"])[:14])
        if pid:
            drug2pert[cid] = pid
            pert2drug[pid] = cid
    return drug2pert, pert2drug


def soragni_pert_map(repo: Path) -> dict[str, str]:
    """pert_id -> Soragni PubChem CID (string) (downloads L1000 pert_info to /tmp)."""
    cache = Path("/tmp/l1000_pert_info.txt.gz")
    if not cache.exists():
        urllib.request.urlretrieve(PERT_INFO_URL, cache)
    pert = pd.read_csv(cache, sep="\t", low_memory=False)
    dr = pd.read_csv(repo / "data/raw/coderdata/sarcoma_drugs.tsv.gz", sep="\t")
    _, ds = build_sample_design(
        load_tranche("sarcoma", repo), "tumor", "viability", drug_key="pubchem_cid"
    )
    soragni_cids = [str(d) for d in ds["drug"]]
    dr_cid = dr["pubchem_id"].map(lambda c: str(int(c)) if pd.notna(c) else None)
    sor = cast("pd.DataFrame", dr[dr_cid.isin(soragni_cids)])
    _, pert2drug = drug_pert_maps(sor, pert)
    return pert2drug


def _drug_of(path: Path, gen: ad.AnnData, valid: set[str]) -> str:
    """Find the L1000 pert_id a generated file corresponds to (Stack writes
    ``generated/<pert_id>.h5ad``)."""
    if path.stem in valid:
        return path.stem
    for tok in path.stem.replace("-", "_").split("_"):
        if tok in valid:
            return tok
    for key in ("pert_id", "condition", "drug"):
        v = gen.uns.get(key) if key in gen.uns else None
        if isinstance(v, str) and v in valid:
            return v
    return ""


def build_generated_deltas(
    generated_dir: Path,
    baseline_path: Path,
    pert_to_drug: dict[str, str],
    *,
    use_logcpm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """delta = generated_treated - baseline per (organoid, drug), gene-aligned.

    Returns ``(delta[pairs x genes], key[patient, drug])``. Files that do not map to
    a drug in ``pert_to_drug`` are skipped (with a note)."""
    base = ad.read_h5ad(baseline_path)
    base_df = pd.DataFrame(
        dense(base.X),
        index=pd.Index([str(o) for o in base.obs_names]),
        columns=pd.Index([str(g) for g in base.var_names]),
    )
    if use_logcpm:
        base_df = logcpm(base_df)
    valid = set(pert_to_drug)
    delta_rows: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    genes: pd.Index | None = None
    for f in sorted(Path(generated_dir).glob("*.h5ad")):
        gen = ad.read_h5ad(f)
        pid = _drug_of(f, gen, valid)
        if not pid:
            print(f"  skip {f.name}: no pert_id match")
            continue
        g = pd.DataFrame(
            dense(gen.X),
            index=pd.Index([str(o) for o in gen.obs_names]),
            columns=pd.Index([str(x) for x in gen.var_names]),
        )
        if use_logcpm:
            g = logcpm(g)
        if genes is None:
            genes = base_df.columns.intersection(g.columns)
        orgs = base_df.index.intersection(g.index)
        d = g.loc[orgs, genes].to_numpy() - base_df.loc[orgs, genes].to_numpy()
        for org, row in zip(orgs, d, strict=True):
            delta_rows.append(row)
            keys.append((str(org), pert_to_drug[pid]))
    if genes is None or not delta_rows:
        raise ValueError("no generated files matched a drug; check generated_dir / mapping")
    delta = pd.DataFrame(np.asarray(delta_rows), columns=genes)
    key = pd.DataFrame(keys, columns=pd.Index(["patient", "drug"]))
    return delta, key


def build_additive_deltas(
    l1000_delta: pd.DataFrame,
    l1000_key: pd.DataFrame,
    patients: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-Stack baseline: each drug's mean L1000 delta, broadcast to every organoid.

    Takes real L1000 treated-minus-DMSO deltas (``build_l1000_gdsc_pairs``) and their
    ``(patient, drug)`` key, averages the delta over cell lines per drug, then assigns
    that single per-drug delta to every organoid in ``patients`` -- so the predicted
    delta is organoid-independent. This is the generation analogue of the drug-mean
    baseline: it carries the drug's main transcriptional effect but no organoid x drug
    interaction, the floor Stack's generated delta must beat. Returns ``(delta[pairs x
    genes], key[patient, drug])`` in the same shape as ``build_generated_deltas``.
    """
    drug_mean = l1000_delta.groupby(l1000_key["drug"].to_numpy()).mean()
    drugs = np.asarray(drug_mean.index, dtype=object)
    pats = np.asarray([str(p) for p in patients], dtype=object)
    n_p = len(pats)
    # each drug's delta repeated once per organoid; keys tile organoids within drug.
    delta = pd.DataFrame(np.repeat(drug_mean.to_numpy(), n_p, axis=0), columns=drug_mean.columns)
    key = pd.DataFrame(
        {"patient": np.tile(pats, len(drugs)), "drug": np.repeat(drugs, n_p)},
    )
    return delta, key


def build_learned_deltas(
    train_base: pd.DataFrame,
    train_delta: pd.DataFrame,
    train_key: pd.DataFrame,
    target_base: pd.DataFrame,
    patients: list[str],
    *,
    reducer: str = "pca",
    k: int = 20,
    alpha: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Organoid-specific delta predictor -- the generation analogue of the expression
    baselines, sitting between the additive floor and Stack.

    Learns a baseline -> delta map on real L1000: reduce each training cell line's DMSO
    baseline by PCA/NMF, regress (ridge) the treated-minus-DMSO delta *residual* (delta
    minus the per-drug mean) on those components, then predict each Soragni organoid's
    correction from its own baseline. The prediction is
    ``delta(organoid, drug) = drug_mean[drug] + correction(organoid)`` -- organoid-
    specific (so it can express within-drug interaction) but driven by a simple linear
    map. The correction transfers across the L1000<->Soragni platform gap in standardized
    units (PCA z-scores per cohort; NMF clips to non-negative). Returns ``(delta[pairs x
    genes], key[patient, drug])`` in the same shape as the other delta sources.
    """
    if reducer not in ("pca", "nmf"):
        raise ValueError("reducer must be 'pca' or 'nmf'")
    g = sorted(
        {str(c) for c in train_base.columns}
        & {str(c) for c in train_delta.columns}
        & {str(c) for c in target_base.columns}
    )
    if not g:
        raise ValueError("no shared genes among train_base, train_delta, target_base")

    drug_mean = train_delta[g].groupby(train_key["drug"].to_numpy()).mean()  # drug x gene
    resid = train_delta[g].to_numpy(dtype=np.float64) - drug_mean.loc[train_key["drug"]].to_numpy(
        dtype=np.float64
    )

    cells = train_base[g]
    k_eff = max(1, min(k, len(cells) - 1, len(g)))
    tgt = np.nan_to_num(target_base.reindex(columns=g).to_numpy(dtype=np.float64))
    if reducer == "nmf":
        # sklearn-stubs mis-types n_components as str; the API takes an int.
        red = NMF(n_components=k_eff, init="nndsvda", random_state=seed, max_iter=2000)  # type: ignore[arg-type]
        z_cell = red.fit_transform(np.maximum(cells.to_numpy(dtype=np.float64), 0.0))
        z_org = red.transform(np.maximum(tgt, 0.0))
    else:
        sc = StandardScaler().fit(cells.to_numpy(dtype=np.float64))
        pca = PCA(n_components=k_eff, random_state=seed)
        z_cell = pca.fit_transform(sc.transform(cells.to_numpy(dtype=np.float64)))
        z_org = pca.transform(sc.transform(tgt))

    z_cell_df = pd.DataFrame(z_cell, index=pd.Index([str(c) for c in cells.index]))
    pair_feat = z_cell_df.reindex([str(p) for p in train_key["patient"]]).to_numpy()
    ok = ~np.isnan(pair_feat).any(axis=1)  # drop pairs whose cell-line baseline is missing
    model = Ridge(alpha=alpha).fit(pair_feat[ok], resid[ok])

    z_org_df = pd.DataFrame(z_org, index=pd.Index([str(o) for o in target_base.index]))
    pats = [str(p) for p in patients]
    z_use = z_org_df.reindex(pats)
    keep = np.atleast_1d(~z_use.isna().to_numpy().any(axis=1))
    pats_keep = [p for p, kp in zip(pats, keep, strict=True) if kp]
    if not pats_keep:
        raise ValueError("no target organoids have a usable baseline")
    correction = model.predict(z_use[keep].to_numpy())  # (n_keep, gene), organoid-specific

    drugs = np.asarray(drug_mean.index, dtype=object)
    n_p = len(pats_keep)
    dm = drug_mean.to_numpy(dtype=np.float64)  # (drug, gene)
    # delta(drug i, organoid j) = drug_mean[i] + correction[j]; rows are drug-major.
    delta_mat = np.repeat(dm, n_p, axis=0) + np.tile(correction, (len(drugs), 1))
    delta = pd.DataFrame(delta_mat, columns=pd.Index(g))
    key = pd.DataFrame(
        {
            "patient": np.tile(np.asarray(pats_keep, dtype=object), len(drugs)),
            "drug": np.repeat(drugs, n_p),
        }
    )
    return delta, key


def build_l1000_gdsc_pairs(
    repo: Path,
    l1000_dir: Path,
    gctx: str,
    *,
    time: float = 24.0,
    chunk: int = 2000,
    treated_cap: int = 8,
    dmso_cap: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Real L1000 treated-minus-DMSO deltas paired with GDSC2 AUC on shared
    (cell line, drug) pairs.

    Returns ``(delta[pairs x gene symbols], key[patient, drug], gdsc_design[patient,
    drug, y], baseline[cell line x gene symbols])`` -- the last is the per-cell-line DMSO
    baseline for the learned delta predictors. Caps replicates and reads the ``.gctx`` in
    column chunks so memory is bounded; ``cmapPy`` is imported here (Alpine only)."""
    from cmapPy.pandasGEXpress.parse_gctx import parse  # type: ignore  # Alpine-only dep

    pert = pd.read_csv(
        l1000_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", low_memory=False
    )
    inst = pd.read_csv(
        l1000_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz", sep="\t", low_memory=False
    )
    gene = pd.read_csv(l1000_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t")
    xg, dg = build_sample_design(load_tranche("gdscv2", repo), "all", "auc", drug_key="pubchem_cid")
    gdr = pd.read_csv(repo / "data/raw/coderdata/gdscv2_drugs.tsv.gz", sep="\t")
    _, pert2drug = drug_pert_maps(gdr, pert)

    gcell = {_norm(c): str(c) for c in xg.index}
    lcell = {_norm(c): str(c) for c in inst["cell_id"].unique()}
    shared = set(gcell) & set(lcell)
    l_ids = [lcell[k] for k in shared]
    l_to_g = {lcell[k]: gcell[k] for k in shared}
    print(f"shared: {len(pert2drug)} drugs, {len(shared)} cell lines")

    drug_ids = list(pert2drug)
    t = inst[
        inst["pert_id"].isin(drug_ids) & inst["cell_id"].isin(l_ids) & (inst["pert_time"] == time)
    ].copy()
    c = inst[
        (inst["pert_iname"] == "DMSO") & inst["cell_id"].isin(l_ids) & (inst["pert_time"] == time)
    ].copy()
    print(
        f"wells: {len(t)} treated + {len(c)} DMSO; "
        f"capping to <= {treated_cap}/(cell,drug), <= {dmso_cap}/cell"
    )
    t = (
        t.sort_values(by="inst_id")  # type: ignore[call-overload]
        .groupby(  # type: ignore[call-overload]
            ["cell_id", "pert_id"], sort=False
        )
        .head(treated_cap)
    )
    c = (
        c.sort_values(by="inst_id")  # type: ignore[call-overload]
        .groupby(  # type: ignore[call-overload]
            "cell_id", sort=False
        )
        .head(dmso_cap)
    )
    print(f"  after cap: {len(t)} treated + {len(c)} DMSO; reading in chunks of {chunk} ...")

    sym = gene.set_index("pr_gene_id")["pr_gene_symbol"].astype(str)
    t_lab = dict(
        zip(t["inst_id"], t["cell_id"].astype(str) + "\t" + t["pert_id"].astype(str), strict=True)
    )
    c_lab = dict(zip(c["inst_id"], c["cell_id"].astype(str), strict=True))

    def group_means(ids: list[str], lab: dict[str, str]) -> pd.DataFrame:
        tot: pd.DataFrame | None = None
        cnt: pd.Series | None = None
        for i in range(0, len(ids), chunk):
            block = parse(gctx, cid=ids[i : i + chunk]).data_df.T  # wells x genes
            block.index = block.index.map(lab)
            s, n = block.groupby(level=0).sum(), block.groupby(level=0).size()
            tot = s if tot is None else tot.add(s, fill_value=0.0)
            cnt = n if cnt is None else cnt.add(n, fill_value=0)
        assert tot is not None and cnt is not None
        return tot.div(cnt, axis=0)

    tmean = group_means(t["inst_id"].tolist(), t_lab)
    dmean = group_means(c["inst_id"].tolist(), c_lab)
    parts = pd.Series(tmean.index).str.split("\t", expand=True)
    cells, perts = parts[0].to_numpy(), parts[1].to_numpy()
    keep = pd.Series(cells).isin(dmean.index).to_numpy()
    tmean, cells, perts = tmean[keep], cells[keep], perts[keep]
    delta = pd.DataFrame(
        tmean.to_numpy() - dmean.reindex(index=cells, columns=tmean.columns).to_numpy(),
        columns=pd.Index([str(sym.get(int(i), "")) for i in tmean.columns]),
    )
    delta = delta.loc[:, [str(col) != "" for col in delta.columns]]
    delta = delta.loc[:, ~pd.Index(delta.columns).duplicated()]
    key = pd.DataFrame(
        {
            "patient": pd.Series(cells).map(l_to_g).to_numpy(),
            "drug": pd.Series(perts).map(pert2drug).to_numpy(),
        }
    )
    # per-cell-line DMSO baseline (gene symbols, GDSC2-name index) for the learned
    # delta predictors; same gene mapping / dedup as the delta.
    base = pd.DataFrame(
        dmean.to_numpy(),
        index=pd.Index([str(l_to_g.get(str(c), str(c))) for c in dmean.index]),
        columns=pd.Index([str(sym.get(int(i), "")) for i in dmean.columns]),
    )
    base = base.loc[:, [str(col) != "" for col in base.columns]]
    base = base.loc[:, ~pd.Index(base.columns).duplicated()]
    return delta, key, cast("pd.DataFrame", dg), base
