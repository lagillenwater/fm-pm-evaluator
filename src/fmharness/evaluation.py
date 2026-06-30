"""Shared evaluation helpers.

Build a sample-level design from a CoderData bundle, run grouped K-fold with a
probe, and score the held-out predictions. Used by the evaluation scripts and
by the controls so they share one code path.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

from fmharness.data.loaders import CoderDataBundle

_MODEL_TYPE = {"organoid": "patient derived organoid", "tumor": "tumor"}


def cpm_bundle(bundle: CoderDataBundle) -> CoderDataBundle:
    """Return a copy of ``bundle`` with expression X as per-million (CPM).

    Prefers raw integer counts when present (GDSC2 keeps them in
    ``layers['raw_counts']``); otherwise X is already count-derived and
    length-free (Soragni CPM) and is renormalized to per-million. This puts
    GDSC2 and Soragni on one shared, length-free normalization -- required for a
    fair cross-substrate comparison, since the native loaders otherwise leave
    GDSC2 on DESeq2 median-of-ratios and Soragni on CPM.
    """
    expr = bundle.expression.copy()
    m = np.asarray(expr.layers.get("raw_counts", expr.X), dtype=np.float64)
    lib = m.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    expr.X = m / lib * 1e6
    return dataclasses.replace(bundle, expression=expr)


def build_sample_design(
    bundle: CoderDataBundle,
    rna_source: str = "all",
    metric: str = "auc",
    drug_key: str = "drug_id",
):
    """Return (sample x gene expression frame, design[patient, drug, y]).

    ``rna_source`` selects one substrate's RNA by model_type, or "all".
    Expression is averaged per patient over its samples of that substrate.
    The design has one row per (patient, drug) with the mean response of the
    chosen metric.

    ``drug_key`` chooses how drugs are identified in the ``drug`` column:
    ``"drug_id"`` (each dataset's native id -- fine within a dataset) or
    ``"pubchem_cid"`` (the canonical cross-dataset key -- required when joining
    GDSC2 to Soragni, whose native drug ids share no namespace). Assays missing
    the chosen key are dropped; multiple ids collapsing to one key are averaged.
    """
    improve_to_patient = {
        str(s.metadata.get("improve_sample_id")): s.patient_id for s in bundle.samples
    }
    improve_to_model = {
        str(s.metadata.get("improve_sample_id")): str(s.metadata.get("model_type"))
        for s in bundle.samples
    }
    sid_to_patient = {s.sample_id: s.patient_id for s in bundle.samples}

    expr = bundle.expression
    obs = [str(s) for s in expr.obs_names]
    if rna_source == "all":
        keep = list(range(len(obs)))
    else:
        target = _MODEL_TYPE[rna_source]
        keep = [i for i, s in enumerate(obs) if improve_to_model.get(s) == target]
    sub = expr[keep]
    x_df = (
        pd.DataFrame(
            np.asarray(sub.X, dtype=np.float64),
            index=pd.Index([improve_to_patient[str(s)] for s in sub.obs_names]),
            columns=pd.Index([str(v) for v in expr.var_names]),
        )
        .groupby(level=0)
        .mean()
    )

    raw_drug = [getattr(x, drug_key) for x in bundle.drug_assays]
    a = pd.DataFrame(
        {
            "patient": [sid_to_patient.get(x.sample_id, x.sample_id) for x in bundle.drug_assays],
            "drug": [None if d is None else str(d) for d in raw_drug],
            "metric": [x.response_metric for x in bundle.drug_assays],
            "y": [x.response_value for x in bundle.drug_assays],
        }
    )
    a = a[(a["metric"] == metric) & a["drug"].notna() & (a["patient"].isin(x_df.index.tolist()))]
    design = a.groupby(["patient", "drug"], as_index=False)["y"].mean()
    return x_df, design


def grouped_cv_predict(
    probe_factory: Callable[[], object],
    x_df: pd.DataFrame,
    design: pd.DataFrame,
    *,
    n_splits: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Grouped K-fold (split by patient). Fit a fresh probe per fold, collect
    held-out (base, residual, prediction). ``probe_factory`` returns a probe
    exposing ``fit(emb, drugs, y, groups)`` and ``predict_parts(emb, drugs)``.
    """
    patients = design["patient"].to_numpy()
    k = min(n_splits, int(design["patient"].nunique()))
    rows: list[dict[str, object]] = []
    for tr, te in GroupKFold(n_splits=k).split(design, groups=patients):
        d_tr, d_te = design.iloc[tr], design.iloc[te]
        probe = probe_factory()
        probe.fit(  # type: ignore[attr-defined]
            x_df.loc[d_tr["patient"]].to_numpy(),
            list(d_tr["drug"]),
            d_tr["y"].to_numpy(),
            groups=list(d_tr["patient"]),
        )
        base, resid = probe.predict_parts(  # type: ignore[attr-defined]
            x_df.loc[d_te["patient"]].to_numpy(), list(d_te["drug"])
        )
        for (_, r), b, rs in zip(d_te.iterrows(), base, resid, strict=True):
            rows.append(
                {
                    "patient": r["patient"],
                    "drug": r["drug"],
                    "y_true": float(r["y"]),
                    "y_pred": float(b + rs),
                    "y_resid": float(rs),
                }
            )
    return pd.DataFrame.from_records(rows)


def _within_drug_corr(preds: pd.DataFrame, true_col: str, pred_col: str, min_n: int = 3) -> float:
    """Pooled within-drug rank correlation of ``true_col`` vs ``pred_col``.

    Ranks both columns inside each drug, centers the ranks, pools across drugs,
    then takes one Pearson correlation. Working within drug removes the drug
    (column) mean, so this is the per-sample signal beyond drug identity.
    """
    ct, cp = [], []
    for _, g in preds.groupby("drug"):
        if len(g) < min_n:
            continue
        rt = g[true_col].rank().to_numpy()
        rp = g[pred_col].rank().to_numpy()
        ct.append(rt - rt.mean())
        cp.append(rp - rp.mean())
    if not ct:
        return float("nan")
    cta, cpa = np.concatenate(ct), np.concatenate(cp)
    if np.std(cta) < 1e-12 or np.std(cpa) < 1e-12:
        return 0.0
    return float(np.asarray(pearsonr(cta, cpa))[0])


def within_drug_rho(preds: pd.DataFrame, pred_col: str = "y_resid", min_n: int = 3) -> float:
    """Within-drug rank correlation of observed vs prediction (``pred_col``).

    Removes the drug mean only. Includes the general-sensitivity effect: an
    organoid sensitive to most drugs ranks low-AUC inside every drug, so this
    rewards predicting that overall sensitivity as well as drug-specific signal.
    """
    return _within_drug_corr(preds, "y_true", pred_col, min_n)


def interaction_rho(preds: pd.DataFrame, pred_col: str = "y_resid", min_n: int = 3) -> float:
    """Drug-specific (organoid x drug interaction) rank correlation.

    Removes each organoid's mean (across its drugs) from both observed and
    predicted before the within-drug correlation, so the general-sensitivity
    effect drops out. What remains is whether the model predicts that an
    organoid responds to *this* drug better or worse than its overall
    sensitivity and the drug's overall potency imply. This is the headline:
    it measures drug-specific signal, the part a shared slope cannot produce.
    """
    p = preds.copy()
    p["_t"] = p["y_true"] - p.groupby("patient")["y_true"].transform("mean")
    p["_p"] = p[pred_col] - p.groupby("patient")[pred_col].transform("mean")
    # A predictor that is constant within an organoid -- e.g. a shared slope,
    # whose only output is one per-organoid offset -- carries no interaction
    # information. After removing the organoid mean, _p is then floating-point
    # dust (~1e-17); ranking it would manufacture a spurious correlation that
    # merely shadows the general-sensitivity signal. Return 0 outright.
    scale = float(np.std(preds[pred_col].to_numpy(dtype=float)))
    if float(np.std(p["_p"].to_numpy(dtype=float))) <= 1e-9 * scale:
        return 0.0
    return _within_drug_corr(p, "_t", "_p", min_n)


def global_spearman(preds: pd.DataFrame) -> float:
    return float(np.asarray(spearmanr(preds["y_true"], preds["y_pred"]))[0])


def regret_norm_at_k(preds: pd.DataFrame, ks: tuple[int, ...] = (1, 3, 5)) -> dict[int, float]:
    """Mean normalized regret@k over patients (lower is better; 0 is best).

    ``y_true`` / ``y_pred`` are AUC-like, so a lower value is a more sensitive (better)
    drug. For each patient the drugs are ranked by ascending ``y_pred`` (predicted best
    first); for the top-k picks, regret is the gap between the best *observed* response
    among them and the patient's actual best, normalized by that patient's observed
    spread so it is panel-size invariant. 0 means the top-k shortlist already contains
    the patient's best drug. Patients with fewer than 2 drugs or no spread are skipped.
    """
    acc: dict[int, list[float]] = {k: [] for k in ks}
    for _, g in preds.groupby("patient"):
        yt = g["y_true"].to_numpy(dtype=np.float64)
        yp = g["y_pred"].to_numpy(dtype=np.float64)
        rng = float(yt.max() - yt.min())
        if len(yt) < 2 or rng <= 0.0:
            continue
        order = np.argsort(yp, kind="stable")  # predicted best (lowest AUC) first
        best = float(yt.min())
        for k in ks:
            topk = order[:k]
            acc[k].append((float(yt[topk].min()) - best) / rng)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between every row of ``a`` (m x g) and ``b`` (n x g),
    returned as an ``m x n`` matrix. Each row is centered and L2-normalized, so the matmul
    of the normalized matrices is the correlation; a constant (zero-variance) row becomes a
    zero vector and therefore correlates 0 with everything (rather than NaN)."""

    def _unit(m: np.ndarray) -> np.ndarray:
        c = m - m.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(c, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        return c / nrm

    return _unit(np.asarray(a, dtype=np.float64)) @ _unit(np.asarray(b, dtype=np.float64)).T


def delta_fidelity(
    pred_delta: pd.DataFrame,
    pred_key: pd.DataFrame,
    real_delta: pd.DataFrame,
    real_key: pd.DataFrame,
    *,
    n_hvg: int | None = 2000,
) -> pd.DataFrame:
    """Faithfulness of a predicted expression delta to the real one, per (patient, drug).

    For every (patient, drug) present in both sources, the Pearson correlation between the
    predicted and the real log-fold-change profile over genes -- Stack's own generation
    metric (predicted vs observed expression *changes*), and the data-level concordance metric
    for pseudobulk-vs-bulk. To expose the failure mode a smooth, non-specific predictor hides
    (every profile correlates with every other), it also reports, per matched pair, the mean
    correlation to the *wrong* real pairs (``r_offdiag``) and the matched pair's specificity
    rank among all real pairs (``rank``; 1.0 = the right pair is the single best match). A
    faithful, specific predictor has ``r`` >> ``r_offdiag`` and ``rank`` ~ 1.

    ``n_hvg`` restricts scoring to the most variable genes of the real delta across the matched
    pairs (mirroring the paper's top-2000 log-normalized HVGs); ``None`` uses all shared genes.
    Returns one row per matched pair: ``patient, drug, r, r_offdiag, rank, n_genes``.
    """
    genes = pred_delta.columns.intersection(real_delta.columns)
    if len(genes) == 0:
        raise ValueError("pred_delta and real_delta share no genes")
    pk, rk = pred_key.reset_index(drop=True), real_key.reset_index(drop=True)
    m = pk.assign(_i=np.arange(len(pk))).merge(
        rk.assign(_j=np.arange(len(rk))), on=["patient", "drug"], how="inner"
    )
    if m.empty:
        raise ValueError("pred and real share no (patient, drug) pairs")
    p = pred_delta[genes].to_numpy(dtype=np.float64)[m["_i"].to_numpy()]
    r = real_delta[genes].to_numpy(dtype=np.float64)[m["_j"].to_numpy()]
    if n_hvg is not None and len(m) > 1 and n_hvg < len(genes):
        top = np.argsort(r.var(axis=0))[::-1][:n_hvg]
        p, r = p[:, top], r[:, top]

    c = _row_corr(p, r)  # (pairs x pairs); the matched correlations are the diagonal
    matched = np.diag(c).copy()
    n = len(matched)
    if n > 1:
        r_offdiag = (c.sum(axis=1) - matched) / (n - 1)
        rank = (c < matched[:, None]).sum(axis=1) / (n - 1)  # frac of wrong pairs below matched
    else:
        r_offdiag = np.full(n, np.nan)
        rank = np.full(n, np.nan)
    return pd.DataFrame(
        {
            "patient": m["patient"].to_numpy(),
            "drug": m["drug"].to_numpy(),
            "r": matched,
            "r_offdiag": r_offdiag,
            "rank": rank,
            "n_genes": int(p.shape[1]),
        }
    )
