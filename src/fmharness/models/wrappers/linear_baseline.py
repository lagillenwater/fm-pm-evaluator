"""Linear baseline adapter: the encoder is identity.

The "embedding" is the densified expression matrix itself; all real work
(per-feature standardization, regularized regression) happens downstream in the
shared probe. Anything more elaborate than this for the baseline (PCA, gene
selection, per-gene z-score) belongs in the probe pipeline, not the adapter.
See ``docs/adapter_contract.md`` §7.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from anndata import AnnData

from fmharness.models.adapter import as_dense_f32
from fmharness.schema import ModelMetadata


class LinearBaselineAdapter:
    """Identity-encoder baseline. Satisfies ``ModelAdapter``."""

    def version(self) -> str:
        return "linear_baseline@v1.0"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus="none",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain="none",
        )

    def embed(self, adata: AnnData) -> np.ndarray:
        # The encoder is identity; the probe's StandardScaler handles
        # per-feature standardization.
        return as_dense_f32(adata)

    def predict_native(self, adata: AnnData, drug_ids: list[str]) -> np.ndarray | None:
        return None
