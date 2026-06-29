"""Tests for the pure Tahoe ingest helpers (the datasets streaming is Alpine-only)."""

from __future__ import annotations

import numpy as np

from fmharness.tahoe import parse_dose_um, scatter_tokens


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


def test_parse_dose_um() -> None:
    assert parse_dose_um("[('8-Hydroxyquinoline',0.05,'uM')]") == 0.05
    assert parse_dose_um("[('Foo',5.0,'uM')]") == 5.0
    assert np.isnan(parse_dose_um("garbage"))
