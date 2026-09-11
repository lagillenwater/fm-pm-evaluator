"""Tests for the pure Tahoe ingest helpers (the datasets streaming is Alpine-only)."""

# scipy ships no PEP-561 type stubs in this environment; under strict mode that turns every
# scipy call site into a cascade of reportUnknown* noise about *its* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import numpy as np

from fmharness.tahoe import parse_dose_um, scatter_flat_tokens, scatter_tokens


def test_scatter_tokens_maps_panel_and_drops_marker_and_offpanel() -> None:
    token_to_col = {10: 0, 20: 1, 30: 2}  # panel genes -> columns
    # first token per cell is the marker (99, off-panel); 40 is an off-panel gene -> both dropped
    genes = [np.array([99, 10, 20]), np.array([99, 30, 40])]
    exprs = [np.array([7.0, 1.0, 2.0]), np.array([7.0, 3.0, 9.0])]
    m = scatter_tokens(genes, exprs, token_to_col, 3)
    assert m.shape == (2, 3)
    assert np.allclose(m.toarray(), [[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])


def test_scatter_tokens_empty() -> None:
    m = scatter_tokens([], [], {10: 0}, 1)
    assert m.shape == (0, 1)


def test_scatter_flat_tokens_matches_scatter_tokens_including_a_zero_panel_gene_cell() -> None:
    """``scatter_tokens`` is a thin wrapper over ``scatter_flat_tokens`` (task 4 review round
    2: the two must not be able to drift apart). This flattens the same ragged cells
    ``scatter_tokens`` is tested against above by hand -- plus a third cell whose tokens are
    all off-panel/marker (zero panel genes) -- and checks the flat-input helper gives exactly
    the same result as the ragged-input wrapper, including that all-zero row."""
    token_to_col = {10: 0, 20: 1, 30: 2}
    genes = [
        np.array([99, 10, 20]),
        np.array([99, 30, 40]),
        np.array([99, 40, 41]),  # no panel genes at all -- must scatter to an all-zero row
    ]
    exprs = [
        np.array([7.0, 1.0, 2.0]),
        np.array([7.0, 3.0, 9.0]),
        np.array([7.0, 5.0, 6.0]),
    ]

    via_wrapper = scatter_tokens(genes, exprs, token_to_col, 3)

    lengths = np.array([len(g) for g in genes], dtype=np.int64)
    flat_tokens = np.concatenate(genes)
    flat_values = np.concatenate(exprs)
    via_flat = scatter_flat_tokens(flat_tokens, flat_values, lengths, token_to_col, 3)

    assert via_flat.shape == via_wrapper.shape == (3, 3)
    np.testing.assert_array_equal(via_flat.toarray(), via_wrapper.toarray())
    assert via_flat.toarray()[2].tolist() == [0.0, 0.0, 0.0]


def test_scatter_flat_tokens_empty() -> None:
    m = scatter_flat_tokens(np.array([], dtype=np.int64), np.array([]), np.array([]), {10: 0}, 1)
    assert m.shape == (0, 1)


def test_parse_dose_um() -> None:
    assert parse_dose_um("[('8-Hydroxyquinoline',0.05,'uM')]") == 0.05
    assert parse_dose_um("[('Foo',5.0,'uM')]") == 5.0
    assert np.isnan(parse_dose_um("garbage"))
