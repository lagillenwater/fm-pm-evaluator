"""Tests for fmharness.models: the ModelAdapter contract + reference adapters.

Covers the Day-7 deliverables (adapter Protocol, MockAdapter, linear baseline)
against the behaviors required by docs/adapter_contract.md §4: deterministic
embeddings, no row reordering, and structural conformance to the Protocol.
"""

from __future__ import annotations

from datetime import date

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from fmharness.models import (
    LinearBaselineAdapter,
    MockAdapter,
    ModelAdapter,
    as_dense_f32,
)
from fmharness.schema import ModelMetadata


def _adata(n_obs: int = 6, n_vars: int = 5, *, seed: int = 1) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_obs, n_vars)).astype(np.float32)
    obs_names = [f"s{i:02d}" for i in range(n_obs)]
    var_names = [f"g{j:02d}" for j in range(n_vars)]
    a = ad.AnnData(X=x)
    a.obs_names = obs_names
    a.var_names = var_names
    return a


# --- Protocol conformance -------------------------------------------------


@pytest.mark.parametrize("adapter", [LinearBaselineAdapter(), MockAdapter()])
def test_satisfies_protocol(adapter: object) -> None:
    assert isinstance(adapter, ModelAdapter)


def test_metadata_is_model_metadata() -> None:
    for adapter in (LinearBaselineAdapter(), MockAdapter()):
        meta = adapter.metadata()
        assert isinstance(meta, ModelMetadata)
        assert meta.pretraining_corpus == "none"
        assert meta.task_signal_in_pretrain == "none"
        assert meta.pretraining_cutoff_date == date(1970, 1, 1)


def test_version_is_stable_string() -> None:
    assert LinearBaselineAdapter().version() == "linear_baseline@v1.0"
    assert MockAdapter(embedding_dim=4, seed=2).version() == "mock@v1.0.0-d4-s2"


# --- Linear baseline ------------------------------------------------------


def test_linear_embed_is_identity() -> None:
    a = _adata()
    emb = LinearBaselineAdapter().embed(a)
    assert emb.dtype == np.float32
    assert emb.shape == (a.n_obs, a.n_vars)
    np.testing.assert_array_equal(emb, np.asarray(a.X, dtype=np.float32))


def test_linear_embed_densifies_sparse() -> None:
    a = _adata()
    dense = np.asarray(a.X, dtype=np.float32)
    a.X = sparse.csr_matrix(dense)
    emb = LinearBaselineAdapter().embed(a)
    np.testing.assert_array_equal(emb, dense)


def test_linear_predict_native_is_none() -> None:
    assert LinearBaselineAdapter().predict_native(_adata(), ["d1", "d2"]) is None


# --- MockAdapter ----------------------------------------------------------


def test_mock_embed_shape() -> None:
    a = _adata(n_obs=6, n_vars=5)
    emb = MockAdapter(embedding_dim=8).embed(a)
    assert emb.shape == (6, 8)
    assert emb.dtype == np.float32


def test_mock_embed_is_deterministic_bitwise() -> None:
    a = _adata()
    adapter = MockAdapter(embedding_dim=8, seed=0)
    first = adapter.embed(a)
    second = adapter.embed(a)
    assert first.tobytes() == second.tobytes()


def test_mock_embed_preserves_row_order() -> None:
    a = _adata(n_obs=6, n_vars=5)
    adapter = MockAdapter(embedding_dim=8, seed=0)
    emb = adapter.embed(a)

    reversed_a = a[::-1].copy()
    emb_reversed = adapter.embed(reversed_a)
    # Row i of the reversed input must map to row i of the reversed output.
    np.testing.assert_array_equal(emb_reversed, emb[::-1])


def test_mock_native_off_by_default() -> None:
    assert MockAdapter().predict_native(_adata(), ["d1", "d2"]) is None


def test_mock_native_on_returns_shape() -> None:
    a = _adata(n_obs=6)
    adapter = MockAdapter(supports_native=True)
    out = adapter.predict_native(a, ["d1", "d2", "d3"])
    assert out is not None
    assert out.shape == (6, 3)


# --- helper ---------------------------------------------------------------


def test_as_dense_f32_rejects_none() -> None:
    a = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    a.X = None
    with pytest.raises(ValueError, match=r"adata\.X is None"):
        as_dense_f32(a)
