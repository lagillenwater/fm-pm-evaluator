"""Tests for the Stack<->Soragni gene panel mapping.

A Stack symbol maps directly to a measured gene's canonical symbol, or by an
unambiguous alias of a measured Entrez id; an Entrez id is never used twice.
"""

from __future__ import annotations

import pandas as pd

from fmharness.stack_panel import build_panel


def _genes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entrez_id": [1, 1, 2, 3, 9],
            "other_id": ["A1B", "AAA", "B2", "OLDC", "N1"],
            "gene_symbol": ["AAA", "AAA", "BBB", "CCC", "NNN"],
            "other_id_source": ["entrez_alias"] * 5,
        }
    )


def test_direct_and_alias_match() -> None:
    # measured = {1, 2, 9}; gene 3 (CCC) is not measured.
    # AAA->1, BBB->2 direct; N1 is an alias of measured gene 9 -> recovered;
    # OLDC is an alias of unmeasured gene 3 -> dropped; ZZZ absent -> dropped.
    panel = build_panel(_genes(), {1, 2, 9}, ["AAA", "BBB", "OLDC", "N1", "ZZZ"])
    mapping = {s: int(e) for s, e in zip(panel["stack_symbol"], panel["entrez_id"], strict=True)}
    assert mapping == {"AAA": 1, "BBB": 2, "N1": 9}
    match = dict(zip(panel["stack_symbol"], panel["match"], strict=True))
    assert match["AAA"] == "symbol"
    assert match["N1"] == "alias"


def test_no_entrez_reused_across_symbol_and_alias() -> None:
    genes = pd.DataFrame(
        {
            "entrez_id": [1, 1],
            "other_id": ["AAA", "OLD_AAA"],
            "gene_symbol": ["AAA", "AAA"],
            "other_id_source": ["entrez_alias", "entrez_alias"],
        }
    )
    # AAA maps directly to gene 1; its alias OLD_AAA must not re-add gene 1.
    panel = build_panel(genes, {1}, ["AAA", "OLD_AAA"])
    assert list(panel["stack_symbol"]) == ["AAA"]
    assert [int(e) for e in panel["entrez_id"]] == [1]


def test_sorted_and_stable() -> None:
    panel = build_panel(_genes(), {1, 2, 9}, ["BBB", "AAA"])
    assert list(panel["stack_symbol"]) == ["AAA", "BBB"]  # sorted by symbol
