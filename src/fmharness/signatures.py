"""Fixed transcriptional signatures for the Path-B viability bridge.

A drug's effect on viability leaves a transcriptional trace: apoptosis / p53
stress genes go up, proliferation genes go down. With no viability-labelled
transcriptome to fit a learned bridge, we score these fixed signatures on a
treated-minus-control expression delta to get a sensitivity proxy. The same
signatures are used to validate the readout on real L1000 deltas (vs GDSC2
viability) and to score Stack-generated deltas (vs Soragni viability), so the
two are directly comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from fmharness.controls import permute_within_drug
from fmharness.evaluation import global_spearman, interaction_rho

# name -> (gene symbols, direction): +1 = up under a working drug => more sensitive;
#                                     -1 = down under a working drug => more sensitive
SIGNATURES: dict[str, tuple[tuple[str, ...], int]] = {
    "apoptosis_p53": (
        ("CDKN1A", "MDM2", "BAX", "BBC3", "PMAIP1", "GADD45A", "SESN1", "SESN2", "BTG2",
         "FAS", "TNFRSF10B", "CASP3", "CASP7", "CASP8", "BID", "BCL2L11", "FDXR", "RRM2B",
         "TP53I3", "DDB2", "XPC", "PHLDA3", "ZMAT3", "CCNG1", "AEN", "TP53INP1"), 1),
    "proliferation": (
        ("MKI67", "CCNB1", "CCNB2", "CCNA2", "CDK1", "CDC20", "TOP2A", "PCNA", "MCM2",
         "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "BUB1", "AURKB", "PLK1", "FOXM1",
         "CENPA", "KIF11", "TYMS", "RRM2", "BIRC5"), -1),
}

# MSigDB Hallmark sets (written by fetch_hallmark.py) as a published alternative to
# the curated SIGNATURES above: p53 / apoptosis induced under a working drug (+1),
# E2F / G2-M proliferation suppressed (-1).
_HALLMARK_DIRECTION: dict[str, int] = {
    "HALLMARK_P53_PATHWAY": 1,
    "HALLMARK_APOPTOSIS": 1,
    "HALLMARK_E2F_TARGETS": -1,
    "HALLMARK_G2M_CHECKPOINT": -1,
}


def load_hallmark(path: str | Path) -> dict[str, tuple[tuple[str, ...], int]]:
    """Load the four Hallmark sets from a GMT, each tagged with its sensitivity
    direction, in the same ``{name: (genes, direction)}`` form as ``SIGNATURES``."""
    sigs: dict[str, tuple[tuple[str, ...], int]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        direction = _HALLMARK_DIRECTION.get(parts[0])
        if direction is not None:
            sigs[parts[0]] = (tuple(g for g in parts[2:] if g), direction)
    return sigs


def _zscore(delta: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (column) across samples; near-constant genes -> scale 1."""
    arr = delta.to_numpy(dtype=np.float64)
    sd = arr.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return pd.DataFrame((arr - arr.mean(axis=0)) / sd, index=delta.index, columns=delta.columns)


def sensitivity_scores(
    delta: pd.DataFrame,
    signatures: dict[str, tuple[tuple[str, ...], int]] | None = None,
) -> pd.DataFrame:
    """Per-sample sensitivity score for each signature, from a treated-minus-control
    expression delta (rows = samples, columns = gene symbols).

    Higher = more sensitive. Each gene's delta is z-scored across samples first so
    the score is not dominated by a few high-variance genes; the score is the
    direction-signed mean over the signature's present genes. Returns one column
    per signature (signatures with no genes present in ``delta`` are dropped).
    ``signatures`` defaults to the curated ``SIGNATURES``; pass ``load_hallmark(...)``
    to score the published Hallmark sets instead.
    """
    sigs = SIGNATURES if signatures is None else signatures
    z = _zscore(delta)
    out: dict[str, pd.Series] = {}
    for name, (genes, direction) in sigs.items():
        present = [g for g in genes if g in z.columns]
        if present:
            out[name] = pd.Series(direction * z[present].to_numpy().mean(axis=1), index=z.index)
    return pd.DataFrame(out, index=delta.index)


def score_signatures(
    delta: pd.DataFrame,
    key: pd.DataFrame,
    design: pd.DataFrame,
    *,
    signatures: dict[str, tuple[tuple[str, ...], int]] | None = None,
    n_perm: int = 1000,
    n_random: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Score each signature against a response (e.g. AUC), with two controls.

    ``delta`` rows align 1:1 with ``key`` rows (columns = gene symbols); ``key``
    has 'patient','drug'; ``design`` has 'patient','drug','y'. The prediction is
    ``-sensitivity`` (AUC-like), scored by interaction rho. Per signature:

    * ``p_label`` -- within-drug label-permutation null (ordering beyond chance);
    * ``rnd_p95`` / ``p_vs_random`` -- the NEGATIVE CONTROL: same-size random gene
      sets scored identically. The signature is specific only if its interaction
      exceeds ``rnd_p95`` (``p_vs_random`` small); otherwise its signal is a
      generic perturbation-magnitude artifact, not death biology.
    """
    z = _zscore(delta)
    base = key.reset_index(drop=True).reset_index().merge(
        design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
    ridx = base["index"].to_numpy()
    drug_s = cast("pd.Series", base["drug"])
    y_s = cast("pd.Series", base["y_true"])
    cols = list(z.columns)
    rng = np.random.default_rng(seed)

    def interaction_of(scorevec: np.ndarray) -> tuple[float, pd.DataFrame]:
        p = pd.DataFrame({
            "patient": base["patient"].to_numpy(),
            "drug": base["drug"].to_numpy(),
            "y_true": base["y_true"].to_numpy(),
            "y_pred": -scorevec[ridx],
        })
        return interaction_rho(p, "y_pred"), p

    sigs = SIGNATURES if signatures is None else signatures
    out: list[dict[str, object]] = []
    for name, (genes, direction) in sigs.items():
        present = [g for g in genes if g in z.columns]
        if not present:
            continue
        it, preds = interaction_of(direction * z[present].to_numpy().mean(axis=1))
        gl = global_spearman(preds)
        lab = np.array([
            interaction_rho(
                preds.assign(y_true=permute_within_drug(
                    drug_s, y_s, np.random.default_rng(seed + 1 + b))),
                "y_pred")
            for b in range(n_perm)
        ])
        rnd = np.array([
            interaction_of(
                direction
                * z[list(rng.choice(cols, len(present), replace=False))].to_numpy().mean(axis=1),
            )[0]
            for _ in range(n_random)
        ])
        out.append({
            "signature": name, "n_genes": len(present), "interaction": round(it, 3),
            "global": round(gl, 3),
            "p_label": round(float(np.mean(lab >= it)), 3),
            "rnd_p95": round(float(np.quantile(rnd, 0.95)), 3),
            "p_vs_random": round(float(np.mean(rnd >= it)), 3),
        })
    return pd.DataFrame(out)
