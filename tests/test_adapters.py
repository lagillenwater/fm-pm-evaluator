"""Tests for the modular viability adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmharness.adapters import (
    ALL_METHODS,
    SignatureAdapter,
    SzalaiLinearAdapter,
    build_adapters,
)

SIGS: dict[str, tuple[tuple[str, ...], int]] = {"death": (("A", "B", "C"), 1)}


def test_build_default_is_all_methods() -> None:
    adapters = build_adapters(signatures=SIGS)
    assert [a.name for a in adapters] == list(ALL_METHODS)
    assert all(a.citation for a in adapters)  # every method carries a citation


def test_build_subset_selects_methods() -> None:
    adapters = build_adapters(["szalai"], signatures=SIGS)
    assert len(adapters) == 1
    assert adapters[0].name == "szalai" and adapters[0].supervised


def test_hallmark_requires_signatures() -> None:
    with pytest.raises(ValueError, match="signatures"):
        build_adapters(["hallmark"])


def test_hallmark_scores_induced_death_most_sensitive() -> None:
    cols = ["A", "B", "C", "N1", "N2"]
    rng = np.random.default_rng(0)
    delta = pd.DataFrame(rng.normal(0, 0.1, (6, 5)), columns=pd.Index(cols),
                         index=pd.Index([f"s{i}" for i in range(6)]))
    delta.loc["s0", ["A", "B", "C"]] += 5.0  # strong death induction in s0
    scores = SignatureAdapter(SIGS).predict(delta)
    assert int(np.argmax(scores)) == 0


def test_szalai_transfers_direction_to_heldout_cohort() -> None:
    rng = np.random.default_rng(1)
    cols = pd.Index(list("abcd"))
    x_tr = pd.DataFrame(rng.normal(size=(80, 4)), columns=cols)
    via_tr = x_tr["a"].to_numpy() * 2 + rng.normal(0, 0.1, 80)  # viability tracks gene a
    adapter = SzalaiLinearAdapter().fit(x_tr, via_tr)
    x_te = pd.DataFrame(rng.normal(size=(40, 4)), columns=cols)
    via_te = x_te["a"].to_numpy() * 2
    sens = adapter.predict(x_te)  # higher = more sensitive = lower viability
    assert float(np.corrcoef(sens, via_te)[0, 1]) < -0.5


def test_xgboost_runs_or_skips_without_libomp() -> None:
    from fmharness.adapters import XGBoostAdapter
    rng = np.random.default_rng(2)
    cols = pd.Index([f"g{i}" for i in range(8)])
    x_tr = pd.DataFrame(rng.normal(size=(60, 8)), columns=cols)
    via_tr = x_tr["g0"].to_numpy() + rng.normal(0, 0.1, 60)
    try:
        adapter = XGBoostAdapter(n_features=5, n_estimators=20).fit(x_tr, via_tr)
    except RuntimeError as e:
        pytest.skip(str(e))  # libomp missing on this machine
    pred = adapter.predict(pd.DataFrame(rng.normal(size=(10, 8)), columns=cols))
    assert pred.shape == (10,)
