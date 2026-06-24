"""Tests for the nonlinear KernelProbe.

Mirrors test_simple.py: n_components=0 is the drug-mean baseline (zero residual),
predict == base + residual, it is deterministic, and an unseen drug falls back to
the global mean. The distinguishing test is quadratic recovery -- the RBF head
captures an even (nonlinear) embedding signal that a linear ridge cannot -- the
unit-level proof that the head has power, the companion to the script-level
positive control.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmharness.probe import KernelProbe, SimpleProbe


def _data(n: int = 80, dim: int = 4, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim))
    drugs = ["drugA" if i % 2 else "drugB" for i in range(n)]
    drug_offset = np.array([2.0 if d == "drugA" else -2.0 for d in drugs])
    y = emb[:, 0] * 1.5 - emb[:, 1] + drug_offset + rng.normal(0, 0.1, n)
    return emb, drugs, y


def _quadratic(n: int = 80, dim: int = 4, *, seed: int = 0):
    # y depends on the embedding only through emb[:,0]**2 -- an even function a
    # linear slope cannot capture (its best linear fit is ~flat).
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim))
    drugs = ["drugA" if i % 2 else "drugB" for i in range(n)]
    drug_offset = np.array([2.0 if d == "drugA" else -2.0 for d in drugs])
    y = emb[:, 0] ** 2 + drug_offset + rng.normal(0, 0.05, n)
    return emb, drugs, y


def test_drug_mean_baseline_has_zero_residual() -> None:
    emb, drugs, y = _data(seed=1)
    probe = KernelProbe(n_components=0).fit(emb, drugs, y)
    base, residual = probe.predict_parts(emb, drugs)
    assert np.allclose(residual, 0.0)
    for d in ("drugA", "drugB"):
        m = np.array([x == d for x in drugs])
        assert np.allclose(base[m], y[m].mean())


def test_predict_parts_sum_to_predict() -> None:
    emb, drugs, y = _data(seed=3)
    probe = KernelProbe(n_components=4).fit(emb, drugs, y)
    base, residual = probe.predict_parts(emb, drugs)
    np.testing.assert_allclose(probe.predict(emb, drugs), base + residual)


def test_kernel_recovers_quadratic_signal_linear_cannot() -> None:
    emb, drugs, y = _quadratic(seed=2)
    split = 60
    kp = KernelProbe(n_components=4).fit(emb[:split], drugs[:split], y[:split])
    sp = SimpleProbe(n_components=4, per_drug=True).fit(emb[:split], drugs[:split], y[:split])
    # score the embedding part (residual) against the centered truth, so the
    # comparison is about the nonlinear term, not the shared drug mean.
    yt = y[split:]
    base_k, res_k = kp.predict_parts(emb[split:], drugs[split:])
    base_s, res_s = sp.predict_parts(emb[split:], drugs[split:])
    corr_k = np.corrcoef(res_k, yt - base_k)[0, 1]
    corr_s = np.corrcoef(res_s, yt - base_s)[0, 1]
    assert corr_k > 0.5
    assert corr_k > corr_s + 0.2


def test_deterministic() -> None:
    emb, drugs, y = _data(seed=4)
    p1 = KernelProbe(n_components=4).fit(emb, drugs, y).predict(emb, drugs)
    p2 = KernelProbe(n_components=4).fit(emb, drugs, y).predict(emb, drugs)
    assert p1.tobytes() == p2.tobytes()


def test_unseen_drug_falls_back_to_global_mean() -> None:
    emb, drugs, y = _data(seed=5)
    probe = KernelProbe(n_components=4).fit(emb, drugs, y)
    base, residual = probe.predict_parts(emb[:3], ["never_seen"] * 3)
    assert np.allclose(base, float(np.mean(y)))
    assert np.allclose(residual, 0.0)  # no per-drug model for an unseen drug


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        KernelProbe().predict(np.zeros((2, 3)), ["d", "d"])


def test_unknown_reducer_raises() -> None:
    with pytest.raises(ValueError, match="reducer"):
        KernelProbe(reducer="svd")


def test_make_head_builds_both_heads() -> None:
    from fmharness.probe import make_head

    emb, drugs, y = _data(seed=6)
    for name, cls in (("linear", SimpleProbe), ("kernel", KernelProbe)):
        probe = make_head(name, n_components=4)()
        assert isinstance(probe, cls)
        probe.fit(emb, drugs, y)
        pred = probe.predict(emb, drugs)
        assert pred.shape == (len(drugs),)
    with pytest.raises(ValueError, match="unknown head"):
        make_head("bilinear")
