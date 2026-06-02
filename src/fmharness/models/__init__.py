"""Model adapters for the harness.

Every model implements the ``ModelAdapter`` Protocol so the pipeline stays
model-agnostic. ``MockAdapter`` is the deterministic test reference; concrete
wrappers live under ``fmharness.models.wrappers``. See
``docs/adapter_contract.md``.
"""

from fmharness.models.adapter import (
    GenePanelMismatch,
    MockAdapter,
    ModelAdapter,
    as_dense_f32,
)
from fmharness.models.wrappers.linear_baseline import LinearBaselineAdapter

__all__ = [
    "GenePanelMismatch",
    "LinearBaselineAdapter",
    "MockAdapter",
    "ModelAdapter",
    "as_dense_f32",
]
