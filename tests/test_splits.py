"""Tests for fmharness.splits.

Property-style tests using a synthetic patient list (mirrors the real
Soragni cohort structure but smaller). The plan's success criterion for
Day 6 is: no (patient, subtype) overlap across train/test for any fold,
on either dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from fmharness.splits import (
    LeavePatientOut,
    LeaveSubtypeOut,
    MissingSplitError,
    SplitFold,
    StratifiedInDistribution,
    require_split,
)


@dataclass
class _P:
    """Minimal stand-in for fmharness.schema.Patient (just the two fields splits care about)."""

    patient_id: str
    subtype: str | None


def _cohort_balanced() -> list[_P]:
    """12 patients across 4 subtypes (3 each) -- safe for 3-fold stratified."""
    pats: list[_P] = []
    for subtype in ["Ewing", "Osteo", "Rhabdo", "Synovial"]:
        for i in range(3):
            pats.append(_P(patient_id=f"{subtype.lower()}-{i:02d}", subtype=subtype))
    return pats


def _cohort_rare_subtype() -> list[_P]:
    """10 patients, one rare subtype with only 1 patient -- triggers the rare-collapse path."""
    pats = [_P(f"osteo-{i:02d}", "Osteo") for i in range(5)]
    pats += [_P(f"ewing-{i:02d}", "Ewing") for i in range(4)]
    pats += [_P("rare-00", "OneOff")]
    return pats


def _no_overlap(fold: SplitFold) -> None:
    assert not (set(fold.train_patient_ids) & set(fold.test_patient_ids)), (
        f"fold {fold.fold_id} has patient_id(s) in both train and test"
    )


def _covers(folds: Iterable[SplitFold], expected_ids: set[str]) -> set[str]:
    seen: set[str] = set()
    for f in folds:
        seen |= set(f.test_patient_ids)
    return seen


# ---------------------------------------------------------------------------
# SplitFold invariants
# ---------------------------------------------------------------------------


def test_splitfold_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="train and test"):
        SplitFold(
            fold_id="bad",
            split_name="x",
            train_patient_ids=("a", "b"),
            test_patient_ids=("b", "c"),
            seed=0,
        )


# ---------------------------------------------------------------------------
# LeavePatientOut
# ---------------------------------------------------------------------------


def test_lpo_one_fold_per_patient() -> None:
    pats = _cohort_balanced()
    folds = list(LeavePatientOut(seed=42).split(pats))
    assert len(folds) == len(pats)
    for f in folds:
        _no_overlap(f)
        assert len(f.test_patient_ids) == 1
        assert len(f.train_patient_ids) == len(pats) - 1
        assert f.seed == 42
        assert f.split_name == "leave_patient_out"


def test_lpo_covers_every_patient_exactly_once() -> None:
    pats = _cohort_balanced()
    folds = list(LeavePatientOut(seed=0).split(pats))
    test_ids = [pid for f in folds for pid in f.test_patient_ids]
    assert sorted(test_ids) == sorted(p.patient_id for p in pats)


def test_lpo_rejects_single_patient() -> None:
    with pytest.raises(ValueError, match=">= 2 patients"):
        list(LeavePatientOut(seed=0).split([_P("only-one", "Solo")]))


# ---------------------------------------------------------------------------
# LeaveSubtypeOut
# ---------------------------------------------------------------------------


def test_lso_one_fold_per_subtype_fine() -> None:
    pats = _cohort_balanced()
    folds = list(LeaveSubtypeOut(seed=0, granularity="fine").split(pats))
    assert len(folds) == 4  # 4 subtypes
    # Each fold's test set is exactly one subtype's patients
    test_subtypes = set()
    for f in folds:
        _no_overlap(f)
        # test patients all share one subtype
        test_pats = [p for p in pats if p.patient_id in f.test_patient_ids]
        labels = {p.subtype for p in test_pats}
        assert len(labels) == 1
        test_subtypes |= labels
    assert test_subtypes == {"Ewing", "Osteo", "Rhabdo", "Synovial"}


def test_lso_no_subtype_overlap_between_train_and_test() -> None:
    """The property the plan calls out specifically: no subtype overlap."""
    pats = _cohort_balanced()
    pat_by_id = {p.patient_id: p for p in pats}
    for f in LeaveSubtypeOut(seed=0, granularity="fine").split(pats):
        train_subtypes = {pat_by_id[i].subtype for i in f.train_patient_ids}
        test_subtypes = {pat_by_id[i].subtype for i in f.test_patient_ids}
        assert not (train_subtypes & test_subtypes), (
            f"fold {f.fold_id}: subtype overlap {train_subtypes & test_subtypes}"
        )


def test_lso_coarse_requires_map() -> None:
    with pytest.raises(ValueError, match="subtype_map"):
        LeaveSubtypeOut(seed=0, granularity="coarse")


def test_lso_coarse_groups_via_map() -> None:
    pats = [
        _P("a", "Embryonal Rhabdomyosarcoma"),
        _P("b", "Alveolar Rhabdomyosarcoma"),
        _P("c", "Osteosarcoma"),
        _P("d", "Osteosarcoma"),
    ]
    mapping = {
        "Embryonal Rhabdomyosarcoma": "Rhabdomyosarcoma",
        "Alveolar Rhabdomyosarcoma": "Rhabdomyosarcoma",
    }
    folds = list(LeaveSubtypeOut(seed=0, granularity="coarse", subtype_map=mapping).split(pats))
    # 2 coarse groups: Rhabdomyosarcoma + Osteosarcoma
    assert len(folds) == 2
    by_held = {f.metadata["held_out_subtype"]: f for f in folds}
    assert "Rhabdomyosarcoma" in by_held
    assert set(by_held["Rhabdomyosarcoma"].test_patient_ids) == {"a", "b"}
    assert set(by_held["Osteosarcoma"].test_patient_ids) == {"c", "d"}


def test_lso_missing_subtype_grouped() -> None:
    pats = [_P("a", "Osteo"), _P("b", "Osteo"), _P("c", None), _P("d", None)]
    folds = list(LeaveSubtypeOut(seed=0).split(pats))
    held_outs = {f.metadata["held_out_subtype"] for f in folds}
    assert "__missing__" in held_outs


def test_lso_rejects_single_subtype() -> None:
    pats = [_P("a", "OnlyOne"), _P("b", "OnlyOne"), _P("c", "OnlyOne")]
    with pytest.raises(ValueError, match=">= 2 distinct subtypes"):
        list(LeaveSubtypeOut(seed=0).split(pats))


# ---------------------------------------------------------------------------
# StratifiedInDistribution
# ---------------------------------------------------------------------------


def test_stratified_no_patient_overlap_across_folds() -> None:
    pats = _cohort_balanced()
    splitter = StratifiedInDistribution(seed=0, n_splits=3)
    folds = list(splitter.split(pats))
    assert len(folds) == 3
    test_seen: list[str] = []
    for f in folds:
        _no_overlap(f)
        test_seen.extend(f.test_patient_ids)
    # Every patient appears in exactly one test fold (K-fold property)
    assert sorted(test_seen) == sorted(p.patient_id for p in pats)


def test_stratified_handles_rare_subtypes() -> None:
    """A subtype with fewer patients than n_splits is collapsed into __rare__,
    so sklearn's StratifiedKFold does not raise."""
    pats = _cohort_rare_subtype()
    folds = list(StratifiedInDistribution(seed=0, n_splits=3).split(pats))
    assert len(folds) == 3
    for f in folds:
        _no_overlap(f)


def test_stratified_deterministic_same_seed() -> None:
    pats = _cohort_balanced()
    a = list(StratifiedInDistribution(seed=42, n_splits=3).split(pats))
    b = list(StratifiedInDistribution(seed=42, n_splits=3).split(pats))
    assert [f.test_patient_ids for f in a] == [f.test_patient_ids for f in b]


def test_stratified_different_seed_gives_different_split() -> None:
    pats = _cohort_balanced()
    a = list(StratifiedInDistribution(seed=1, n_splits=3).split(pats))
    b = list(StratifiedInDistribution(seed=2, n_splits=3).split(pats))
    # At least one fold should differ
    assert [f.test_patient_ids for f in a] != [f.test_patient_ids for f in b]


def test_stratified_rejects_too_few_patients() -> None:
    with pytest.raises(ValueError, match="cannot do"):
        list(
            StratifiedInDistribution(seed=0, n_splits=5).split(
                [_P(f"p{i}", "Osteo") for i in range(3)]
            )
        )


def test_stratified_rejects_n_splits_below_2() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        StratifiedInDistribution(seed=0, n_splits=1)


# ---------------------------------------------------------------------------
# require_split guard
# ---------------------------------------------------------------------------


def test_require_split_rejects_none() -> None:
    with pytest.raises(MissingSplitError, match="refuses to run"):
        require_split(None)


def test_require_split_accepts_valid_splitter() -> None:
    splitter = LeavePatientOut(seed=0)
    assert require_split(splitter) is splitter


def test_require_split_rejects_missing_name() -> None:
    class _Fake:
        name = ""
        seed = 0

        def split(self, _patients: list[_P]) -> list[SplitFold]:
            return []

    with pytest.raises(MissingSplitError, match="no ``name``"):
        require_split(_Fake())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cross-splitter sanity: every splitter satisfies no-train-test-overlap on a
# realistic cohort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "splitter_factory",
    [
        lambda: LeavePatientOut(seed=0),
        lambda: LeaveSubtypeOut(seed=0, granularity="fine"),
        lambda: StratifiedInDistribution(seed=0, n_splits=3),
    ],
)
def test_no_patient_overlap_any_splitter(splitter_factory) -> None:
    pats = _cohort_balanced()
    for fold in splitter_factory().split(pats):
        _no_overlap(fold)
        # Every test patient is a real patient
        ids = {p.patient_id for p in pats}
        assert set(fold.test_patient_ids) <= ids
        assert set(fold.train_patient_ids) <= ids
