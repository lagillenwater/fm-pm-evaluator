"""Shared evaluation helpers.

Build a sample-level design from a CoderData bundle, run grouped K-fold with a
probe, and score the held-out predictions. Used by the evaluation scripts and
by the controls so they share one code path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

from fmharness.data.loaders import CoderDataBundle

_MODEL_TYPE = {"organoid": "patient derived organoid", "tumor": "tumor"}


def build_sample_design(bundle: CoderDataBundle, rna_source: str = "all", metric: str = "auc"):
    """Return (sample x gene expression frame, design[patient, drug, y]).

    ``rna_source`` selects one substrate's RNA by model_type, or "all".
    Expression is averaged per patient over its samples of that substrate.
    The design has one row per (patient, drug) with the mean response of the
    chosen metric.
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

    a = pd.DataFrame(
        {
            "patient": [sid_to_patient.get(x.sample_id, x.sample_id) for x in bundle.drug_assays],
            "drug": [x.drug_id for x in bundle.drug_assays],
            "metric": [x.response_metric for x in bundle.drug_assays],
            "y": [x.response_value for x in bundle.drug_assays],
        }
    )
    a = a[(a["metric"] == metric) & (a["patient"].isin(x_df.index.tolist()))]
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


def _within_drug_corr(
    preds: pd.DataFrame, true_col: str, pred_col: str, min_n: int = 3
) -> float:
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
