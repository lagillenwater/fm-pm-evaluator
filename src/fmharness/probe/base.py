"""Shared internals for the probe heads.

A probe head predicts drug response from an embedding as ``a_d + f(z_s)``: a
per-drug mean ``a_d`` plus a residual that some function of the (reduced)
embedding adds on top. Every head shares the same scaffolding -- the per-drug
mean, the PCA/NMF reduction of the embedding, and the input checks -- and differs
only in the function fit to the residual (a ridge slope in ``SimpleProbe``, an
RBF kernel ridge in ``KernelProbe``). That common scaffolding lives here so the
heads stay apples-to-apples: identical reducer, identical drug-mean base, so a
Stack-vs-expression comparison reflects the representation, not the plumbing.

``predict_parts`` (defined per head) returns ``(a_d, residual)`` separately so the
metrics can score the embedding part alone, avoiding the leave-one-out drug-base
artifact.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler

# Penalty path searched by the inner CV. PCA scores of standardized expression
# carry large variance (top components ~10^3), so the path must reach well above
# that for the slope to shrink to ~0 when the embedding is uninformative.
ALPHAS = tuple(float(a) for a in np.logspace(0.0, 8.0, 9))

# A per-drug fit needs a few organoids; below this the drug contributes no
# embedding term (it falls back to its drug mean).
MIN_DRUG_N = 4


class _FlooredScaler(BaseEstimator, TransformerMixin):
    """Standardize, but never divide by a standard deviation below ``floor``.

    A gene nearly constant across the training organoids has a tiny SD; dividing
    by it turns a small expression difference in a held-out organoid into a huge
    standardized value that the slope cannot rein in (the source of the AUC ~1400
    blow-ups). Flooring the SD bounds that amplification while leaving
    well-varying genes essentially untouched. ``floor=0`` reproduces a plain
    StandardScaler (zero-variance genes still get scale 1).
    """

    def __init__(self, floor: float = 0.0) -> None:
        self.floor = floor

    def fit(self, x: ArrayLike, y: object = None) -> _FlooredScaler:
        arr = np.asarray(x, dtype=np.float64)
        self.mean_ = arr.mean(axis=0)
        scale = np.maximum(arr.std(axis=0), self.floor)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: ArrayLike) -> NDArray[np.float64]:
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_


class ProbeBase:
    """Per-drug mean + a PCA/NMF reduction shared by every probe head.

    Subclasses implement the residual model. They call ``_prepare_fit`` from
    ``fit`` to validate inputs, learn the per-drug means, and get back the
    embedding, drug ids, residual, and the reduction rank ``k`` (0 means the
    drug-mean baseline -- no embedding term).
    """

    def __init__(
        self,
        *,
        n_components: int = 10,
        per_drug: bool = False,
        reducer: str = "pca",
        std_floor: float = 0.0,
        seed: int = 0,
    ) -> None:
        if reducer not in ("pca", "nmf"):
            raise ValueError("reducer must be 'pca' or 'nmf'")
        self.n_components = n_components
        self.per_drug = per_drug
        self.reducer = reducer
        self.std_floor = std_floor
        self.seed = seed
        self._drug_means: dict[str, float] = {}
        self._global_mean = 0.0

    def _base(self, drug_ids: Sequence[str]) -> NDArray[np.float64]:
        """Per-drug mean, with the global mean for drugs unseen at fit."""
        return (
            pd.Series(drug_ids)
            .map(self._drug_means)
            .fillna(self._global_mean)
            .to_numpy(dtype=np.float64)
        )

    def _reducer_steps(self, k: int) -> list[tuple[str, BaseEstimator]]:
        """Steps reducing the embedding to k components for the residual model.

        PCA acts on standardized features, so it is not dominated by a few
        high-variance genes and its components are orthogonal. NMF requires
        non-negative input, so it acts on the expression directly and returns
        parts-based, non-negative factors (gene programs) rather than orthogonal
        components -- a more biologically natural low-rank summary of expression.
        """
        if self.reducer == "nmf":
            # sklearn-stubs mis-types n_components as str; the API takes an int.
            nmf = NMF(n_components=k, init="nndsvda", random_state=self.seed, max_iter=2000)  # type: ignore[arg-type]
            return [("nmf", nmf)]
        scaler = _FlooredScaler(self.std_floor) if self.std_floor > 0 else StandardScaler()
        return [
            ("scaler", scaler),
            ("pca", PCA(n_components=k, random_state=self.seed)),
        ]

    def _prepare_fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.object_], NDArray[np.float64], int]:
        """Validate inputs, learn per-drug means, return (emb, drugs, residual, k)."""
        emb = np.asarray(embeddings, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if emb.ndim != 2 or len(drug_ids) != emb.shape[0] or len(y_arr) != emb.shape[0]:
            raise ValueError("embeddings, drug_ids, y must share the same number of rows")
        if self.reducer == "nmf" and emb.size and float(emb.min()) < 0.0:
            raise ValueError("nmf reducer requires non-negative input (e.g. log1p expression)")

        drug_arr = np.asarray(drug_ids, dtype=object)
        self._global_mean = float(y_arr.mean())
        self._drug_means = {
            str(k): float(v) for k, v in pd.Series(y_arr).groupby(drug_arr).mean().items()
        }
        residual = y_arr - self._base(drug_ids)
        # k capped by the rank ceiling; 0 -> drug-mean baseline (no embedding term).
        k = min(self.n_components, max(0, emb.shape[0] - 1), emb.shape[1])
        return emb, drug_arr, residual, k
