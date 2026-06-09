"""Tests for the viability-bridge signatures.

A sample with apoptosis/p53 genes induced should score as most sensitive; a
sample with proliferation genes suppressed should too (opposite direction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.signatures import SIGNATURES, load_hallmark, sensitivity_scores


def test_apoptosis_induction_scores_sensitive() -> None:
    genes = list(SIGNATURES["apoptosis_p53"][0])[:10]
    cols = [*genes, "NOISE1", "NOISE2"]
    rng = np.random.default_rng(0)
    delta = pd.DataFrame(rng.normal(0, 0.1, (6, len(cols))),
                         columns=pd.Index(cols), index=pd.Index([f"s{i}" for i in range(6)]))
    delta.loc["s0", genes] += 5.0  # strong apoptosis induction
    sc = sensitivity_scores(delta)
    assert "apoptosis_p53" in sc.columns
    assert sc["apoptosis_p53"].idxmax() == "s0"


def test_proliferation_suppression_scores_sensitive() -> None:
    genes = list(SIGNATURES["proliferation"][0])[:10]
    rng = np.random.default_rng(1)
    delta = pd.DataFrame(rng.normal(0, 0.1, (6, len(genes))),
                         columns=pd.Index(genes), index=pd.Index([f"s{i}" for i in range(6)]))
    delta.loc["s0", genes] -= 5.0  # proliferation shut down -> sensitive
    sc = sensitivity_scores(delta)
    assert sc["proliferation"].idxmax() == "s0"


def test_absent_signature_dropped() -> None:
    delta = pd.DataFrame({"FOO": [1.0, 2.0], "BAR": [3.0, 4.0]})
    assert sensitivity_scores(delta).shape[1] == 0


def test_load_hallmark_sets() -> None:
    repo = Path(__file__).resolve().parent.parent
    sigs = load_hallmark(repo / "data/static/hallmark_signatures.gmt")
    assert set(sigs) == {"HALLMARK_P53_PATHWAY", "HALLMARK_APOPTOSIS",
                         "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT"}
    p53_genes, p53_dir = sigs["HALLMARK_P53_PATHWAY"]
    assert p53_dir == 1 and "CDKN1A" in p53_genes
    assert sigs["HALLMARK_G2M_CHECKPOINT"][1] == -1  # proliferation, suppressed


def test_sensitivity_scores_accepts_custom_signatures() -> None:
    cols = ["A", "B", "C"]
    delta = pd.DataFrame(np.zeros((4, 3)), columns=pd.Index(cols),
                         index=pd.Index([f"s{i}" for i in range(4)]))
    delta.loc["s0", ["A", "B"]] += 5.0
    sc = sensitivity_scores(delta, {"mysig": (("A", "B"), 1)})
    assert list(sc.columns) == ["mysig"]
    assert sc["mysig"].idxmax() == "s0"
