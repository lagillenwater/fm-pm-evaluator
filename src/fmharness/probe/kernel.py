"""Nonlinear probe: per-drug mean plus an RBF kernel-ridge residual on a few PCs.

Same shape as ``SimpleProbe`` -- ``y(s, d) = a_d + f_d(z_s)`` -- but the residual
``f_d`` is an RBF kernel ridge rather than a linear slope, so the embedding can
matter nonlinearly. This exists to test whether the project's headline result
(a Stack embedding gives no advantage over PCA of expression) is *head-invariant*:
the linear ``SimpleProbe`` might under-read a representation that only pays off
under a nonlinear map. Run the same comparison through this head and the result
should hold across head families.

To stay apples-to-apples with the linear head, ``KernelProbe`` reuses the exact
PCA/NMF reduction from ``probe.base.ProbeBase`` (identical reduced scores), then
standardizes those scores so the RBF width is on a common scale, and fits one
``KernelRidge(kernel="rbf")`` per drug. ``(alpha, gamma)`` are chosen per drug by
leave-one-out ``GridSearchCV``; a large alpha shrinks the residual toward 0, so an
uninformative embedding degrades gracefully to the drug mean -- the same fairness
guarantee ``SimpleProbe`` has. ``KernelRidge`` is closed-form and deterministic.

``predict_parts`` returns ``(a_d, residual)`` separately, matching the linear head
so the metrics score the embedding part alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fmharness.probe.base import MIN_DRUG_N, ProbeBase

__all__ = ["KernelProbe"]

# Regularization path for the kernel ridge. Scores are standardized before the
# kernel, so this is a modest path; the top reaches well past unit scale so the
# residual shrinks to ~0 (drug-mean baseline) when the embedding is uninformative.
_KERNEL_ALPHAS = tuple(float(a) for a in np.logspace(-2.0, 3.0, 6))

# RBF widths, searched per drug. Scores enter standardized, so a fixed grid is
# meaningful across representations.
_GAMMAS = tuple(float(g) for g in np.logspace(-2.0, 0.5, 6))


class KernelProbe(ProbeBase):
    """Per-drug mean + an RBF KernelRidge residual (shared or per-drug) on top-k PCs."""

    def __init__(
        self,
        *,
        n_components: int = 10,
        per_drug: bool = True,
        reducer: str = "pca",
        std_floor: float = 0.0,
        alphas: Sequence[float] = _KERNEL_ALPHAS,
        gammas: Sequence[float] = _GAMMAS,
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
        self.gammas = tuple(gammas)
        # A shared (reducer + score-scaler) transform plus one fitted estimator
        # per drug (per-drug slope), or a single estimator (shared slope).
        self._transform: Pipeline | None = None
        self._models: dict[str, KernelRidge] = {}
        self._shared: KernelRidge | None = None

    def _param_grid(self) -> dict[str, list[float]]:
        return {"alpha": list(self.alphas), "gamma": list(self.gammas)}

    def _search(
        self, scores: NDArray[np.float64], y: NDArray[np.float64], cv: object
    ) -> KernelRidge:
        # neg-MSE scoring works on single-sample LOO folds, where R^2 is undefined.
        gs = GridSearchCV(
            KernelRidge(kernel="rbf"),
            self._param_grid(),
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        gs.fit(scores, y)
        best = gs.best_estimator_
        assert isinstance(best, KernelRidge)
        return best

    def fit(
        self,
        embeddings: ArrayLike,
        drug_ids: Sequence[str],
        y: ArrayLike,
        groups: Sequence[str] | None = None,
    ) -> KernelProbe:
        emb, drug_arr, residual, k = self._prepare_fit(embeddings, drug_ids, y)
        self._transform = None
        self._models = {}
        self._shared = None
        if k > 0:
            # Standardize the PCA/NMF scores so the RBF width is on a common scale.
            self._transform = Pipeline(
                [*self._reducer_steps(k), ("score_scaler", StandardScaler())]
            )
            scores = np.asarray(self._transform.fit_transform(emb), dtype=np.float64)
            if self.per_drug:
                self._fit_per_drug(scores, drug_arr, residual)
            else:
                self._fit_shared(scores, residual, groups)
        return self

    def _fit_per_drug(
        self,
        scores: NDArray[np.float64],
        drug_arr: NDArray[np.object_],
        residual: NDArray[np.float64],
    ) -> None:
        # One RBF kernel ridge per drug, fit on that drug's organoids. Each
        # organoid appears once per drug, so leave-one-out picks (alpha, gamma)
        # without leakage. Drugs with too few organoids get no model (fall back
        # to the drug mean).
        for d in np.unique(drug_arr):
            m = drug_arr == d
            if int(m.sum()) < MIN_DRUG_N:
                continue
            self._models[str(d)] = self._search(scores[m], residual[m], LeaveOneOut())

    def _fit_shared(
        self,
        scores: NDArray[np.float64],
        residual: NDArray[np.float64],
        groups: Sequence[str] | None,
    ) -> None:
        # Penalty/width chosen by sample-grouped CV so the embedding term is
        # judged on held-out samples, mirroring SimpleProbe's shared path.
        cv: object = KFold(n_splits=min(5, len(residual)))
        if groups is not None:
            g = np.asarray(groups)
            n_g = len(np.unique(g))
            if n_g >= 2:
                cv = list(GroupKFold(n_splits=min(5, n_g)).split(scores, residual, groups=g))
        self._shared = self._search(scores, residual, cv)

    def predict_parts(
        self, embeddings: ArrayLike, drug_ids: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self._drug_means:
            raise RuntimeError("probe is not fitted; call fit() before predict()")
        base = self._base(drug_ids)
        emb = np.asarray(embeddings, dtype=np.float64)
        if self._transform is None:  # drug-mean baseline
            return base, np.zeros(len(drug_ids), dtype=np.float64)
        scores = np.asarray(self._transform.transform(emb), dtype=np.float64)
        if self._shared is not None:  # shared slope
            residual = np.asarray(self._shared.predict(scores), dtype=np.float64)
        else:  # per-drug slope: predict each drug's rows in one block; rest -> 0
            residual = np.zeros(len(drug_ids), dtype=np.float64)
            drugs = np.asarray(drug_ids, dtype=object)
            for d, est in self._models.items():
                m = drugs == d
                if m.any():
                    residual[m] = est.predict(scores[m])
        return base, residual

    def predict(self, embeddings: ArrayLike, drug_ids: Sequence[str]) -> NDArray[np.float64]:
        base, residual = self.predict_parts(embeddings, drug_ids)
        return base + residual
