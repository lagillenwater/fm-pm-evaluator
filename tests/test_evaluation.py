"""Tests for shared evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.evaluation import delta_fidelity, regret_norm_at_k, score_predictions


def test_regret_norm_at_k() -> None:
    # y_true / y_pred are AUC-like (lower = better). Patient A's predicted ranking puts
    # its true-best drug first (regret 0); patient B's puts it last (regret 1 at k=1,
    # 0 once k covers all 3). Patients with no spread are skipped.
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "drug": ["d1", "d2", "d3", "d1", "d2", "d3", "d1", "d2"],
            "y_true": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 5.0, 5.0],  # C flat -> skipped
            "y_pred": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0],
        }
    )
    r = regret_norm_at_k(preds, ks=(1, 3))
    assert np.isclose(r[1], 0.5)  # A: 0, B: (30-10)/(30-10)=1 -> mean 0.5
    assert np.isclose(r[3], 0.0)  # top-3 covers every drug for both A and B


def test_delta_fidelity_matches_specific_and_flags_nonspecific() -> None:
    genes = pd.Index(list("abcde"))
    real = pd.DataFrame(
        [
            [3.0, 1.0, -1.0, -2.0, -1.0],  # (P1, d1)
            [-1.0, -2.0, 3.0, 1.0, -1.0],  # (P2, d1) -- a different response shape
        ],
        columns=genes,
    )
    key = pd.DataFrame({"patient": ["P1", "P2"], "drug": ["d1", "d1"]})

    # specific predictor: each pair predicts its own real delta -> matched r = 1, rank = 1,
    # and the matched correlation beats the correlation to the wrong pair.
    spec = delta_fidelity(real.copy(), key.copy(), real, key, n_hvg=None)
    assert np.allclose(spec["r"].to_numpy(), 1.0)
    assert np.allclose(spec["rank"].to_numpy(), 1.0)
    assert (spec["r"].to_numpy() > spec["r_offdiag"].to_numpy()).all()

    # non-specific predictor: BOTH pairs predict the same profile (P1's real delta). P2's
    # matched correlation is then no better than its correlation to the wrong (P1) pair,
    # so its specificity rank collapses -- the smooth-generator failure mode is caught.
    pred = pd.DataFrame([real.iloc[0].to_numpy(), real.iloc[0].to_numpy()], columns=genes)
    nonspec = delta_fidelity(pred, key.copy(), real, key, n_hvg=None)
    p2 = nonspec[nonspec["patient"] == "P2"].iloc[0]
    assert p2["rank"] == 0.0
    assert p2["r"] <= p2["r_offdiag"] + 1e-9


def test_delta_fidelity_restricts_to_hvgs() -> None:
    # only genes a, b vary across the two pairs; c, d, e are constant -> top-2 HVGs = a, b.
    genes = pd.Index(list("abcde"))
    real = pd.DataFrame(np.eye(2, 5) * 3.0 + 1.0, columns=genes)
    key = pd.DataFrame({"patient": ["P1", "P2"], "drug": ["d", "d"]})
    out = delta_fidelity(real.copy(), key, real, key, n_hvg=2)
    assert (out["n_genes"] == 2).all()


def test_score_predictions_reports_interaction_and_null() -> None:
    # perfect predictions (y_pred == y_true) -> interaction rho = 1; the within-drug
    # permutation null almost never reaches 1, so p_label is small.
    rng = np.random.default_rng(0)
    y = rng.normal(size=15)
    preds = pd.DataFrame(
        {
            "patient": [f"P{i}" for i in range(5) for _ in range(3)],
            "drug": ["d1", "d2", "d3"] * 5,
            "y_true": y,
            "y_pred": y,
        }
    )
    s = score_predictions(preds, n_perm=200, seed=0)
    assert s["interaction"] == 1.0
    assert 0.0 <= s["p_label"] <= 0.2
    assert s["n"] == 15.0
    assert "regret@1" in s and "global" in s
