"""Viability adapters: map a treated-minus-control transcriptome delta to a
per-sample drug-sensitivity score (higher = more sensitive). Three published
approaches, selected with ``build_adapters(methods=...)``; the default is all three.

- ``"hallmark"`` -- fixed signature scoring: the direction-signed mean of z-scored
  signature genes (apoptosis / p53 up, proliferation down), averaged across sets.
  No training. MSigDB Hallmark gene sets (Liberzon et al., Cell Systems 2015);
  single-sample scoring in the spirit of ssGSEA (Barbie et al., Nature 2009).
- ``"szalai"`` -- L2-regularized linear regression from the delta to viability, fit
  on real perturbation->viability pairs and applied to the target delta. Szalai et
  al., "Signatures of cell death and proliferation in perturbation transcriptomics
  data -- from confounding factor to effective prediction", Nucleic Acids Research 2019.
- ``"xgboost"`` -- elastic-net gene selection + gradient-boosted trees, fit on
  perturbation->viability pairs. Lu, Chen & Qin, "Drug-induced cell viability prediction
  from LINCS-L1000 through WRFEN-XGBoost algorithm", BMC Bioinformatics 2021.

The supervised adapters (szalai, xgboost) are trained on a perturbation->viability
cohort (e.g. real L1000 deltas vs GDSC2 AUC) and applied to a held-out delta (e.g.
Stack-generated organoid deltas). Each cohort is z-scored by its own per-gene
statistics, so ``predict`` must be given a cohort (many samples), not one row, and
the learned coefficients transfer across the platform gap in standardized units.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

ALL_METHODS: tuple[str, ...] = ("hallmark", "szalai", "xgboost")


class ViabilityAdapter(Protocol):
    """A predictor of per-sample sensitivity (higher = more sensitive) from a delta."""

    name: str
    citation: str
    supervised: bool

    def fit(self, delta: pd.DataFrame, viability: np.ndarray) -> ViabilityAdapter: ...
    def predict(self, delta: pd.DataFrame) -> np.ndarray: ...


def _zscore(delta: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (column) by this cohort's own mean / std."""
    arr = delta.to_numpy(dtype=np.float64)
    sd = arr.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return pd.DataFrame((arr - arr.mean(axis=0)) / sd, index=delta.index, columns=delta.columns)


class SignatureAdapter:
    """Fixed signature scoring (no training); the signed signature means, combined."""

    name = "hallmark"
    supervised = False
    citation = "Liberzon et al., Cell Systems 2015 (Hallmark); Barbie et al., Nature 2009 (ssGSEA)"

    def __init__(self, signatures: dict[str, tuple[tuple[str, ...], int]]) -> None:
        self._sigs = signatures

    def fit(self, delta: pd.DataFrame, viability: np.ndarray) -> SignatureAdapter:
        del delta, viability  # unsupervised
        return self

    def predict(self, delta: pd.DataFrame) -> np.ndarray:
        z = _zscore(delta)
        parts: list[np.ndarray] = []
        for genes, direction in self._sigs.values():
            present = [g for g in genes if g in z.columns]
            if present:
                parts.append(direction * z[present].to_numpy().mean(axis=1))
        if not parts:
            return np.zeros(len(delta), dtype=np.float64)
        return np.asarray(parts, dtype=np.float64).mean(axis=0)


class SzalaiLinearAdapter:
    """L2 linear regression delta -> viability (Szalai et al., Nucleic Acids Research 2019)."""

    name = "szalai"
    supervised = True
    citation = "Szalai et al., Nucleic Acids Research 2019"

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._genes: list[str] = []
        self._model: Any = None

    def fit(self, delta: pd.DataFrame, viability: np.ndarray) -> SzalaiLinearAdapter:
        self._genes = [str(c) for c in delta.columns]
        self._model = Ridge(alpha=self._alpha).fit(
            _zscore(delta).to_numpy(), np.asarray(viability, dtype=np.float64))
        return self

    def predict(self, delta: pd.DataFrame) -> np.ndarray:
        z = _zscore(delta.reindex(columns=self._genes, fill_value=0.0)).to_numpy()
        return -np.asarray(self._model.predict(z), dtype=np.float64)  # higher = more sensitive


class XGBoostAdapter:
    """Elastic-net gene selection + gradient-boosted trees (WRFEN-XGBoost; Wang et al.,
    BMC Bioinformatics 2020). ``xgboost`` is imported lazily (it needs OpenMP/libomp)."""

    name = "xgboost"
    supervised = True
    citation = "Lu, Chen & Qin, BMC Bioinformatics 2021 (WRFEN-XGBoost)"

    def __init__(self, n_features: int = 300, n_estimators: int = 400,
                 max_depth: int = 4, seed: int = 0) -> None:
        self._n_features = n_features
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._seed = seed
        self._genes: list[str] = []
        self._model: Any = None

    def fit(self, delta: pd.DataFrame, viability: np.ndarray) -> XGBoostAdapter:
        try:
            import xgboost as xgb
        except (ImportError, OSError, ValueError) as e:  # ValueError: XGBoostError (libomp)
            raise RuntimeError("xgboost unavailable (on macOS: `brew install libomp`)") from e
        z = _zscore(delta).to_numpy()
        y = np.asarray(viability, dtype=np.float64)
        # elastic-net key-gene selection before boosting (WRFEN selects genes first)
        coef = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,
                          random_state=self._seed).fit(z, y).coef_
        nonzero = int(np.sum(coef != 0))
        k = min(self._n_features, nonzero) if nonzero else min(self._n_features, z.shape[1])
        idx = np.argsort(-np.abs(coef))[:k]
        self._genes = [str(delta.columns[i]) for i in idx]
        self._model = xgb.XGBRegressor(
            n_estimators=self._n_estimators, max_depth=self._max_depth,
            learning_rate=0.05, subsample=0.8, random_state=self._seed).fit(z[:, idx], y)
        return self

    def predict(self, delta: pd.DataFrame) -> np.ndarray:
        z = _zscore(delta.reindex(columns=self._genes, fill_value=0.0)).to_numpy()
        return -np.asarray(self._model.predict(z), dtype=np.float64)


def build_adapters(
    methods: list[str] | None = None,
    *,
    signatures: dict[str, tuple[tuple[str, ...], int]] | None = None,
) -> list[ViabilityAdapter]:
    """Construct the selected viability adapters (default: all of ``ALL_METHODS``).

    ``signatures`` is required when "hallmark" is among the methods.
    """
    chosen = list(ALL_METHODS) if methods is None else methods
    out: list[ViabilityAdapter] = []
    for m in chosen:
        if m == "hallmark":
            if signatures is None:
                raise ValueError("the hallmark adapter requires signatures=")
            out.append(SignatureAdapter(signatures))
        elif m == "szalai":
            out.append(SzalaiLinearAdapter())
        elif m == "xgboost":
            out.append(XGBoostAdapter())
        else:
            raise ValueError(f"unknown method {m!r}; choose from {ALL_METHODS}")
    return out
