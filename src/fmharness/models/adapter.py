"""``ModelAdapter`` Protocol + the reference ``MockAdapter``.

Every model -- linear baseline, STACK, Tahoe-x1, and the metadata-only control
-- implements ``ModelAdapter`` so the rest of the pipeline (splits, probe,
metrics, registry, leakage scan) stays model-agnostic. The contract has three
required surfaces (``embed``, ``metadata``, ``version``) and one optional one
(``predict_native``). See ``docs/adapter_contract.md``.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
from anndata import AnnData

from fmharness.schema import ModelMetadata


class GenePanelMismatch(ValueError):
    """Input gene panel does not match the reference the adapter expects.

    Gene-panel reconciliation is the loader's responsibility, not the
    adapter's: an adapter must refuse mismatched input loudly rather than
    silently aligning it (adapter_contract.md §4.5).
    """


def as_dense_f32(adata: AnnData) -> np.ndarray:
    """Densify ``adata.X`` to a contiguous ``float32`` array, row-aligned with
    ``adata.obs_names``. Handles both dense and sparse backings.
    """
    x = adata.X
    if x is None:
        raise ValueError("adata.X is None; adapters require a populated expression matrix")
    to_dense = getattr(x, "toarray", None)  # scipy sparse / lazy backed arrays
    if to_dense is not None:
        x = to_dense()
    return np.ascontiguousarray(x, dtype=np.float32)


@runtime_checkable
class ModelAdapter(Protocol):
    """Wrap a foundation model (or baseline) so the harness can use it."""

    def version(self) -> str:
        """Stable identifier, e.g. ``tahoe_x1@v1.0.0``. Embedded in PredictionRecord."""
        ...

    def metadata(self) -> ModelMetadata:
        """Pretraining provenance for the leakage scan."""
        ...

    def embed(self, adata: AnnData) -> np.ndarray:
        """Encode samples (rows of ``adata``) into a dense embedding matrix.

        Returns an array of shape ``(adata.n_obs, embedding_dim)``. Rows are
        aligned with ``adata.obs_names``; adapters must not reorder rows.
        """
        ...

    def predict_native(self, adata: AnnData, drug_ids: list[str]) -> np.ndarray | None:
        """Optional native drug-aware head. Return ``None`` if the adapter does
        not expose one. Shape: ``(adata.n_obs, len(drug_ids))``.
        """
        ...


class MockAdapter:
    """Deterministic stand-in adapter for tests.

    ``embed`` projects expression through a seeded, frozen random projection:
    deterministic across calls (contract §4.1) and order-preserving (§4.2),
    because row ``i`` of the output is a function only of row ``i`` of the
    input and the fixed projection. Set ``supports_native=True`` to exercise
    the native-head path; otherwise ``predict_native`` returns ``None``.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        seed: int = 0,
        *,
        supports_native: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.supports_native = supports_native

    def version(self) -> str:
        return f"mock@v1.0.0-d{self.embedding_dim}-s{self.seed}"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            pretraining_corpus="none",
            pretraining_cutoff_date=date(1970, 1, 1),
            task_signal_in_pretrain="none",
        )

    def embed(self, adata: AnnData) -> np.ndarray:
        x = as_dense_f32(adata)
        rng = np.random.default_rng(self.seed)
        projection = rng.standard_normal((x.shape[1], self.embedding_dim)).astype(np.float32)
        return np.ascontiguousarray(x @ projection, dtype=np.float32)

    def predict_native(self, adata: AnnData, drug_ids: list[str]) -> np.ndarray | None:
        if not self.supports_native:
            return None
        rng = np.random.default_rng(self.seed + 1)
        return rng.random((adata.n_obs, len(drug_ids)), dtype=np.float32)
