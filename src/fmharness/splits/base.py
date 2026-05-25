"""Split fold model + Splitter protocol + ``require_split`` guard.

Splits operate on **patients**, not samples or assays. Each fold gives the set
of patient_ids that belong to train vs test; downstream code (Day-7
Evaluator, Day-9 probe) maps those patient_ids to sample_ids / assay rows via
the Tranche bundle's ``Sample`` and ``DrugAssay`` lists.

The seed is recorded on every ``SplitFold`` so it flows verbatim into the
``EnvironmentSnapshot`` attached to each downstream ``PredictionRecord``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SplitFold:
    """One fold of a split. Train and test are disjoint sets of patient_ids."""

    fold_id: str
    split_name: str
    train_patient_ids: tuple[str, ...]
    test_patient_ids: tuple[str, ...]
    seed: int
    # Free-form metadata recorded alongside the fold (e.g., the subtype that
    # was held out by LeaveSubtypeOut). Persisted into PredictionRecord on
    # downstream runs so the report can stratify by it.
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        train_set = set(self.train_patient_ids)
        test_set = set(self.test_patient_ids)
        if train_set & test_set:
            overlap = sorted(train_set & test_set)
            raise ValueError(
                f"SplitFold {self.fold_id!r} has patient_id(s) in BOTH train and test: {overlap}"
            )


@runtime_checkable
class SplittablePatient(Protocol):
    """The two attributes a splitter cares about. ``Patient`` satisfies it."""

    patient_id: str
    subtype: str | None


@runtime_checkable
class Splitter(Protocol):
    """Every splitter implements this."""

    name: str
    seed: int

    def split(self, patients: Sequence[SplittablePatient]) -> Iterator[SplitFold]: ...


class MissingSplitError(ValueError):
    """Raised when downstream code (Evaluator, probe) is invoked without a named split."""


def require_split(splitter: Splitter | None) -> Splitter:
    """Guard for downstream code: refuse to proceed without a named splitter.

    Plan §7 (Day 6): ``Evaluator`` refuses to run without a named split.
    Downstream callers do ``splitter = require_split(splitter)`` at their
    entry point so the harness never silently runs unsplit predictions.
    """
    if splitter is None:
        raise MissingSplitError(
            "no splitter provided -- the harness refuses to run unsplit predictions. "
            "Pass an instance of StratifiedInDistribution, LeavePatientOut, or "
            "LeaveSubtypeOut."
        )
    name = getattr(splitter, "name", None)
    if not isinstance(name, str) or not name:
        raise MissingSplitError(
            f"splitter {splitter!r} has no ``name`` attribute -- every splitter "
            "must declare one so prediction records can record split_name."
        )
    return splitter
