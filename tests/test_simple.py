"""Tests for the minimal SimpleProbe.

n_components=0 is the drug-mean baseline (residual is zero). With components it
learns a shared slope that recovers an embedding signal. predict == base +
residual, it is deterministic, and an unseen drug falls back to the global mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmharness.probe import SimpleProbe


def _data(n: int = 120, dim: int = 6, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim))
    drugs = ["drugA" if i % 2 else "drugB" for i in range(n)]
    drug_offset = np.array([2.0 if d == "drugA" else -2.0 for d in drugs])
    y = emb[:, 0] * 1.5 - emb[:, 1] + drug_offset + rng.normal(0, 0.1, n)
    return emb, drugs, y


def test_drug_mean_baseline_has_zero_residual() -> None:
    emb, drugs, y = _data(seed=1)
    probe = SimpleProbe(n_components=0).fit(emb, drugs, y)
    base, residual = probe.predict_parts(emb, drugs)
    assert np.allclose(residual, 0.0)
    # base is the per-drug mean
    for d in ("drugA", "drugB"):
        m = np.array([x == d for x in drugs])
        assert np.allclose(base[m], y[m].mean())


def test_linear_recovers_embedding_signal() -> None:
    emb, drugs, y = _data(seed=2)
    split = 90
    probe = SimpleProbe(n_components=6).fit(emb[:split], drugs[:split], y[:split])
    pred = probe.predict(emb[split:], drugs[split:])
    assert np.corrcoef(pred, y[split:])[0, 1] > 0.9


def test_predict_parts_sum_to_predict() -> None:
    emb, drugs, y = _data(seed=3)
    probe = SimpleProbe(n_components=4).fit(emb, drugs, y)
    base, residual = probe.predict_parts(emb, drugs)
    np.testing.assert_allclose(probe.predict(emb, drugs), base + residual)


def test_deterministic() -> None:
    emb, drugs, y = _data(seed=4)
    p1 = SimpleProbe(n_components=5).fit(emb, drugs, y).predict(emb, drugs)
    p2 = SimpleProbe(n_components=5).fit(emb, drugs, y).predict(emb, drugs)
    assert p1.tobytes() == p2.tobytes()


def test_unseen_drug_falls_back_to_global_mean() -> None:
    emb, drugs, y = _data(seed=5)
    probe = SimpleProbe(n_components=0).fit(emb, drugs, y)
    base, _ = probe.predict_parts(emb[:3], ["never_seen"] * 3)
    assert np.allclose(base, float(np.mean(y)))


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        SimpleProbe().predict(np.zeros((2, 3)), ["d", "d"])


def test_nmf_reducer_recovers_signal() -> None:
    # Non-negative embedding; NMF factors should let the shared slope recover a
    # linear response, just as PCA does.
    rng = np.random.default_rng(7)
    n, dim = 120, 6
    emb = rng.random((n, dim)) * 5.0  # non-negative, required by NMF
    drugs = ["drugA" if i % 2 else "drugB" for i in range(n)]
    offset = np.array([1.0 if d == "drugA" else -1.0 for d in drugs])
    y = emb[:, 0] * 2.0 - emb[:, 1] + offset + rng.normal(0, 0.1, n)
    split = 90
    probe = SimpleProbe(n_components=6, reducer="nmf").fit(emb[:split], drugs[:split], y[:split])
    pred = probe.predict(emb[split:], drugs[split:])
    assert np.corrcoef(pred, y[split:])[0, 1] > 0.7


def test_nmf_rejects_negative_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SimpleProbe(reducer="nmf").fit(-np.ones((10, 3)), ["d"] * 10, np.zeros(10))


def test_unknown_reducer_raises() -> None:
    with pytest.raises(ValueError, match="reducer"):
        SimpleProbe(reducer="svd")


def test_std_floor_tames_near_constant_gene() -> None:
    # A gene near-constant in training has a tiny SD; standardizing by it sends a
    # held-out organoid that differs on that gene to a huge value and the
    # prediction blows up. The variance floor must keep it bounded.
    rng = np.random.default_rng(11)
    n, dim = 60, 4
    emb = rng.standard_normal((n, dim))
    emb[:, 0] = 1.0 + rng.normal(0, 1e-4, n)  # gene 0 ~constant in training
    drugs = ["a"] * n
    y = emb[:, 1] + rng.normal(0, 0.1, n)
    test = emb[:1].copy()
    test[0, 0] = 5.0  # held-out organoid: large jump on the near-constant gene
    base = float(np.mean(y))
    no = SimpleProbe(n_components=dim, std_floor=0.0, alphas=(1.0,)).fit(emb, drugs, y)
    fl = SimpleProbe(n_components=dim, std_floor=0.5, alphas=(1.0,)).fit(emb, drugs, y)
    dev_no = abs(float(no.predict(test, ["a"])[0]) - base)
    dev_fl = abs(float(fl.predict(test, ["a"])[0]) - base)
    assert dev_no > dev_fl  # floor keeps the held-out prediction closer to the data
