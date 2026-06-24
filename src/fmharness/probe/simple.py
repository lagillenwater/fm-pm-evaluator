"""Minimal linear probe: per-drug mean plus a ridge slope on a few PCs.

This is the simplest model that can rank samples within a drug from an embedding:

    y(s, d) = a_d + z_s . b        (shared slope, ``per_drug=False``)
    y(s, d) = a_d + z_s . b_d       (per-drug slope, ``per_drug=True``)

``a_d`` is the per-drug mean response (with the global mean for a drug unseen at
fit). The embedding term is a slope on the top-k components of ``z_s`` -- PCA of
the standardized embedding (``reducer="pca"``) or non-negative gene programs
(``reducer="nmf"``) -- fit by ``RidgeCV`` so the penalty is chosen by CV and the
slope shrinks toward 0 when the embedding is uninformative; the model then
degrades gracefully to the drug mean rather than injecting noise.

The shape of the slope decides what the model can predict:

* A **shared** slope ``b`` adds the same offset to every drug for a given
  organoid, so it can only express *general sensitivity* (an organoid broadly
  more/less responsive). It is structurally unable to predict a drug-specific
  (organoid x drug) interaction.
* A **per-drug** slope ``b_d`` lets the same expression direction matter
  differently for different drugs, so the prediction can vary across organoids
  *within* a drug -- i.e. it can predict drug response, not just sensitivity.
  Each drug's slope is fit on its own organoids over the shared PCA scores.

Set ``n_components=0`` to drop the embedding term entirely: the probe then
predicts ``a_d`` for every sample, i.e. the drug-mean baseline.

The embedding comes from an adapter: raw expression for the linear baseline, a
foundation-model vector for Stack. Swapping the embedding is the only change, so
the same probe scores every model.

``predict_parts`` returns ``(a_d, residual)`` separately so the metrics can score
the embedding part alone, avoiding the leave-one-out drug-base artifact. The
per-drug mean and the PCA/NMF reduction shared with the other heads live in
``probe.base.ProbeBase``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from fmharness.probe.base import ALPHAS, MIN_DRUG_N, ProbeBase

__all__ = ["SimpleProbe"]


class SimpleProbe(ProbeBase):
    """Per-drug mean + a RidgeCV slope (shared or per-drug) on the top-k PCs."""

    def __init__(
        self,
        *,
        n_components: int = 10,
        per_drug: bool = False,
        reducer: str = "pca",
        std_floor: float = 0.0,
        alphas: Sequence[float] = ALPHAS,
        seed: int = 0,
    ) -> None:
        super().__init__(
            n_components=n_components,
            per_drug=per_drug,
            reducer=reducer,
            std_floor=std_floor,
            seed=seed,
        )
        self.alphas = tuple(alphas)
        # shared-slope state
        self._embed: Pipeline | None = None
        # per-drug-slope state: a shared (scaler, pca) transform plus one
        # (coef, intercept) per drug, stacked for vectorized prediction.
        self._transform: Pipeline | None = None
        self._coef: pd.DataFrame | None = None  # index=drug, cols=PCs
        self._intercept: pd.Series | None = None  # index=drug

    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> SimpleProbe:
        emb, drug_arr, residual, k = self._prepare_fit(embeddings, drug_ids, y)
        self._embed = self._transform = None
        self._coef = self._intercept = None
        if k > 0:
            if self.per_drug:
                self._fit_per_drug(emb, drug_arr, residual, k)
            else:
                self._fit_shared(emb, residual, k, groups)
        return self

    def _fit_shared(
        self,
        emb: NDArray[np.float64],
        residual: NDArray[np.float64],
        k: int,
        groups: Sequence[str] | None,
    ) -> None:
        # Choose the ridge penalty by sample-grouped CV so the per-sample
        # embedding term is judged on held-out samples, not rows of a sample
        # already in training. Otherwise the penalty is picked too weak and the
        # slope adds noise on truly held-out samples.
        cv = None
        if groups is not None:
            g = np.asarray(groups)
            n_g = len(np.unique(g))
            if n_g >= 2:
                cv = list(GroupKFold(n_splits=min(5, n_g)).split(emb, residual, groups=g))
        # asarray, not the stored tuple: RidgeCV's leave-one-out path mutates
        # alphas in place, which fails on an immutable tuple when cv is None.
        ridge = RidgeCV(alphas=np.asarray(self.alphas, dtype=np.float64), cv=cv)
        self._embed = Pipeline([*self._reducer_steps(k), ("ridge", ridge)])
        self._embed.fit(emb, residual)

    def _fit_per_drug(
        self,
        emb: NDArray[np.float64],
        drug_arr: NDArray[np.object_],
        residual: NDArray[np.float64],
        k: int,
    ) -> None:
        # One shared representation (scaler + PCA), then a separate ridge slope
        # per drug fit on that drug's organoids. Each organoid appears once per
        # drug, so RidgeCV's default leave-one-out picks the penalty without
        # leakage. Drugs with too few organoids get no slope (fall back to mean).
        self._transform = Pipeline(self._reducer_steps(k))
        scores = self._transform.fit_transform(emb)
        coef: dict[str, NDArray[np.float64]] = {}
        intercept: dict[str, float] = {}
        for d in np.unique(drug_arr):
            m = drug_arr == d
            if int(m.sum()) < MIN_DRUG_N:
                continue
            ridge = RidgeCV(alphas=np.asarray(self.alphas, dtype=np.float64)).fit(
                scores[m], residual[m]
            )
            coef[str(d)] = np.asarray(ridge.coef_, dtype=np.float64)
            intercept[str(d)] = float(ridge.intercept_)
        if coef:
            self._coef = pd.DataFrame.from_dict(coef, orient="index")
            self._intercept = pd.Series(intercept)

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self._drug_means:
            raise RuntimeError("probe is not fitted; call fit() before predict()")
        base = self._base(drug_ids)
        emb = np.asarray(embeddings, dtype=np.float64)
        if self._embed is not None:  # shared slope
            residual = np.asarray(self._embed.predict(emb), dtype=np.float64)
        elif self._transform is not None and self._coef is not None:  # per-drug slope
            scores = np.asarray(self._transform.transform(emb), dtype=np.float64)
            # Map each row's drug to its slope; unseen / too-rare drugs -> 0.
            assert self._intercept is not None
            b = self._coef.reindex(list(drug_ids)).fillna(0.0).to_numpy(dtype=np.float64)
            c = self._intercept.reindex(list(drug_ids)).fillna(0.0).to_numpy(dtype=np.float64)
            residual = (scores * b).sum(axis=1) + c
        else:  # drug-mean baseline
            residual = np.zeros(len(drug_ids), dtype=np.float64)
        return base, residual

    def predict(self, embeddings: ArrayLike, drug_ids: Sequence[str]) -> NDArray[np.float64]:
        base, residual = self.predict_parts(embeddings, drug_ids)
        return base + residual
