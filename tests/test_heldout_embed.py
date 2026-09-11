"""Task 6 of rung 1: Stack embeddings -- the pure, locally testable pieces.

Design.md section 4 fixes what a Stack line description is (the mean embedding of a line's
DMSO cells) and section 8 fixes the build step's checks: a description built from half a
line's cells should best match that line's other half (and fall to chance once cells' line
labels are shuffled), and the drug fine-tune's weights must differ from the base checkpoint's.
Every test below runs the real, shipped functions -- ``scripts/heldout_embed.py`` (loaded by
path, since ``scripts`` is not a package and none of this needs torch, anndata, or ``stack``)
and ``scripts/strip_ckpt_head.py``.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmharness.heldout.descriptions import group_means, identity_match, shuffled_identity_null

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


he = _load_module("heldout_embed", "scripts/heldout_embed.py")
sch = _load_module("strip_ckpt_head", "scripts/strip_ckpt_head.py")

pytestmark = pytest.mark.known_answer


# ==============================================================================================
# line_means
# ==============================================================================================


def test_line_means_hand_checked_and_half_multiindex() -> None:
    # Two lines, two dims. Line A: 3 cells (halves 0, 0, 1); line B: 2 cells (halves 0, 1).
    cell_embeddings = np.array(
        [
            [1.0, 10.0],  # A, half 0
            [3.0, 30.0],  # A, half 0
            [5.0, 50.0],  # A, half 1
            [2.0, 20.0],  # B, half 0
            [4.0, 40.0],  # B, half 1
        ]
    )
    lines = np.array(["A", "A", "A", "B", "B"])
    halves = np.array([0, 0, 1, 0, 1])

    per_line, per_half = he.line_means(cell_embeddings, lines, halves)

    assert list(per_line.index) == ["A", "B"]
    assert per_line.index.name == "line"
    assert list(per_line.columns) == ["dim_0", "dim_1"]
    np.testing.assert_allclose(per_line.loc["A"].to_numpy(), [3.0, 30.0])  # mean(1,3,5)=3
    np.testing.assert_allclose(per_line.loc["B"].to_numpy(), [3.0, 30.0])  # mean(2,4)=3

    assert isinstance(per_half.index, pd.MultiIndex)
    assert per_half.index.names == ["line", "half"]
    np.testing.assert_allclose(per_half.loc[("A", 0)].to_numpy(), [2.0, 20.0])  # mean(1,3)
    np.testing.assert_allclose(per_half.loc[("A", 1)].to_numpy(), [5.0, 50.0])
    np.testing.assert_allclose(per_half.loc[("B", 0)].to_numpy(), [2.0, 20.0])
    np.testing.assert_allclose(per_half.loc[("B", 1)].to_numpy(), [4.0, 40.0])


# ==============================================================================================
# encoders_differ
# ==============================================================================================


def test_encoders_differ_identical_dicts() -> None:
    state = {"a": np.array([1.0, 2.0]), "b": np.array([[1, 2], [3, 4]])}
    result = he.encoders_differ(state, {k: v.copy() for k, v in state.items()})
    assert result == {
        "n_shared": 2,
        "n_identical": 2,
        "n_different": 0,
        "identical_all": True,
        "only_in_a": [],
        "only_in_b": [],
    }


def test_encoders_differ_one_differing_tensor() -> None:
    state_a = {"a": np.array([1.0, 2.0]), "b": np.array([1.0, 1.0])}
    state_b = {"a": np.array([1.0, 2.0]), "b": np.array([9.0, 9.0])}
    result = he.encoders_differ(state_a, state_b)
    assert result["n_shared"] == 2
    assert result["n_identical"] == 1
    assert result["n_different"] == 1
    assert result["identical_all"] is False
    assert result["only_in_a"] == []
    assert result["only_in_b"] == []


def test_encoders_differ_disjoint_keys() -> None:
    state_a = {"only_a": np.array([1.0])}
    state_b = {"only_b": np.array([2.0])}
    result = he.encoders_differ(state_a, state_b)
    assert result["n_shared"] == 0
    assert result["n_identical"] == 0
    assert result["n_different"] == 0
    assert result["identical_all"] is False  # nothing shared, so never "identical"
    assert result["only_in_a"] == ["only_a"]
    assert result["only_in_b"] == ["only_b"]


def test_encoders_differ_keep_filters_before_comparing() -> None:
    state_a = {"model.encoder.w": np.array([1.0]), "model.cls.0.weight": np.array([1.0])}
    state_b = {"model.encoder.w": np.array([1.0]), "model.cls.0.weight": np.array([9.0])}
    # Excluding the head key, the only compared key ("encoder.w") is identical on both sides.
    result = he.encoders_differ(state_a, state_b, keep=lambda k: not k.endswith("cls.0.weight"))
    assert result["n_shared"] == 1
    assert result["identical_all"] is True


# ==============================================================================================
# head_keys (scripts/strip_ckpt_head.py)
# ==============================================================================================


def test_head_keys_selects_only_generation_head_suffixes() -> None:
    keys = [
        "model.encoder.blocks.0.attn.weight",
        "model.encoder.blocks.0.attn.bias",
        "model.query_pos_embedding",
        "model.cls.0.weight",
        "model.cls.0.bias",
        "model.cls.2.weight",
        "model.cls.2.bias",
        "model.encoder.norm.weight",
    ]
    removed = sch.head_keys(keys)
    assert removed == [
        "model.query_pos_embedding",
        "model.cls.0.weight",
        "model.cls.0.bias",
        "model.cls.2.weight",
        "model.cls.2.bias",
    ]


def test_head_keys_no_gen_head_keys_present() -> None:
    keys = ["model.encoder.blocks.0.attn.weight", "model.encoder.norm.weight"]
    assert sch.head_keys(keys) == []


def test_head_keys_matches_bare_suffix_with_no_prefix() -> None:
    assert sch.head_keys(["query_pos_embedding", "cls.0.weight"]) == [
        "query_pos_embedding",
        "cls.0.weight",
    ]


# ==============================================================================================
# embedding identity control (known answer): identity share 1.0 for distinct-centre synthetic
# embeddings, above the group_means null's 99th percentile; shuffled labels give chance.
# ==============================================================================================


def _synthetic_cell_embeddings(
    n_lines: int, width: int, cells_per_line: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic per-cell Stack-like embeddings for ``n_lines`` lines with distinct per-line
    centres (well-separated so each line is trivially identifiable from its own cells),
    halves assigned at random. Fully vectorized (no per-cell or per-line Python loop)."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-50.0, 50.0, size=(n_lines, width))
    line_idx = np.repeat(np.arange(n_lines), cells_per_line)
    noise = rng.normal(scale=0.5, size=(line_idx.shape[0], width))
    embeddings = centres[line_idx] + noise
    halves = rng.integers(0, 2, size=line_idx.shape[0]).astype(np.int64)
    lines_arr = np.array([f"L{i}" for i in line_idx])
    return embeddings, lines_arr, halves


@pytest.mark.step_build
def test_embedding_identity_recovered() -> None:
    n_lines = 50
    embeddings, lines_arr, halves = _synthetic_cell_embeddings(
        n_lines=n_lines, width=8, cells_per_line=40, seed=0
    )

    def describe(x: np.ndarray) -> np.ndarray:
        return x

    categories = sorted(set(lines_arr.tolist()))
    half0_labels, half0 = group_means(embeddings[halves == 0], lines_arr[halves == 0])
    half1_labels, half1 = group_means(embeddings[halves == 1], lines_arr[halves == 1])
    half0 = (
        pd.DataFrame(half0, index=pd.Index(half0_labels))
        .reindex(categories, fill_value=0.0)
        .to_numpy()
    )
    half1 = (
        pd.DataFrame(half1, index=pd.Index(half1_labels))
        .reindex(categories, fill_value=0.0)
        .to_numpy()
    )

    real_share = identity_match(describe(half0), describe(half1))
    assert real_share == pytest.approx(1.0)

    null = shuffled_identity_null(
        embeddings, lines_arr, halves, describe, n_shuffles=50, seed=1, aggregate=group_means
    )
    p99 = float(np.quantile(null, 0.99))
    assert real_share > p99, f"real share {real_share} not above null p99 {p99}"


@pytest.mark.step_build
def test_embedding_shuffled_identity_is_chance() -> None:
    n_lines = 50
    embeddings, lines_arr, halves = _synthetic_cell_embeddings(
        n_lines=n_lines, width=6, cells_per_line=30, seed=2
    )

    def describe(x: np.ndarray) -> np.ndarray:
        return x

    n_shuffles = 50
    null = shuffled_identity_null(
        embeddings, lines_arr, halves, describe, n_shuffles, seed=3, aggregate=group_means
    )
    se = float(null.std(ddof=1)) / np.sqrt(n_shuffles)
    tol = 3.0 * se
    expected = 1.0 / n_lines
    assert abs(float(null.mean()) - expected) <= tol, (
        f"null mean {null.mean()} not within 3 SE ({tol}) of chance {expected}"
    )
