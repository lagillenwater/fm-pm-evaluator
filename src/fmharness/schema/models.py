"""Model-provenance model: ``ModelMetadata``.

Returned by every ``ModelAdapter.metadata()`` and consumed by the leakage scan
to reason about pretraining exposure. ``model_weights_hash`` and
``container_digest`` are required for foundation-model adapters (and captured
into ``EnvironmentSnapshot``) but legitimately ``None`` for baselines, so the
schema keeps them optional; the per-adapter contract enforces the FM case.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Whether the pretraining corpus contained drug-response labels (the task
# signal). ``adjacent`` covers related-but-not-identical signal (e.g.,
# perturbation response without the harness's specific drug-response readout).
TaskSignal = Literal["none", "adjacent", "direct"]


class ModelMetadata(BaseModel):
    """Pretraining provenance for one model adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    pretraining_corpus: str = Field(min_length=1)
    pretraining_cutoff_date: date
    task_signal_in_pretrain: TaskSignal
    model_weights_hash: str | None = None
    container_digest: str | None = None
