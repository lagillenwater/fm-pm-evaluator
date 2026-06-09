"""Tests for the controls.

permute_within_drug must preserve each drug's mean and break the
expression-response link. plant_response must produce a signal the probe can
recover, rising with the effect size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.controls import permute_within_drug, plant_response
from fmharness.evaluation import grouped_cv_predict, within_drug_rho
from fmharness.probe import SimpleProbe


def test_permute_preserves_drug_means() -> None:
    rng = np.random.default_rng(0)
    drugs = pd.Series(["a"] * 10 + ["b"] * 10)
    y = pd.Series(rng.standard_normal(20))
    y_perm = permute_within_drug(drugs, y, rng)
    for d in ["a", "b"]:
        m = (drugs == d).to_numpy()
        assert np.isclose(y[m].mean(), y_perm[m].mean())
        assert sorted(y[m]) == sorted(y_perm[m])  # same values, reordered


def test_plant_signal_rises_with_effect() -> None:
    # Plant a known direction; the probe should recover more within-drug signal
    # as the effect grows, and ~0 at effect 0.
    rng = np.random.default_rng(1)
    n_per, dim, n_drugs = 80, 8, 4
    emb = rng.standard_normal((n_per * n_drugs, dim))
    drugs = pd.Series([f"d{d}" for d in range(n_drugs) for _ in range(n_per)])
    base_y = pd.Series(np.repeat(np.arange(n_drugs, dtype=float), n_per))
    x_df = pd.DataFrame(emb, index=[f"s{i}" for i in range(len(drugs))])  # type: ignore[arg-type]

    def factory():
        return SimpleProbe(n_components=dim)

    rhos = {}
    for eff in (0.0, 2.0):
        y = plant_response(drugs, base_y, emb, effect=eff, rng=np.random.default_rng(5))
        design = pd.DataFrame({"patient": list(x_df.index), "drug": list(drugs), "y": y})
        preds = grouped_cv_predict(factory, x_df, design, n_splits=4, seed=0)
        rhos[eff] = within_drug_rho(preds)
    assert rhos[2.0] > rhos[0.0] + 0.2
    assert abs(rhos[0.0]) < 0.2
