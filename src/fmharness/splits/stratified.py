"""Stratified K-fold CV split: in-distribution baseline.

Stratifies on the patient subtype (``Patient.subtype``) so each fold has a
similar subtype mix to the full cohort. Patients with no subtype are
grouped under ``"__missing__"`` and stratified alongside the others.
"""

from __future__ import annotations

from collections.abc import Iterator

from sklearn.model_selection import StratifiedKFold

from fmharness.splits.base import SplitFold, SplittablePatient

_MISSING_SUBTYPE = "__missing__"


class StratifiedInDistribution:
    """K-fold cross-validation stratified by ``Patient.subtype``."""

    name = "stratified_in_distribution"

    def __init__(self, *, seed: int, n_splits: int = 5) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2; got {n_splits}")
        self.seed = seed
        self.n_splits = n_splits

    def split(self, patients: list[SplittablePatient]) -> Iterator[SplitFold]:
        if len(patients) < self.n_splits:
            raise ValueError(f"cannot do {self.n_splits}-fold split on {len(patients)} patients")
        ids = [p.patient_id for p in patients]
        labels = [p.subtype if p.subtype is not None else _MISSING_SUBTYPE for p in patients]

        # If some subtype has fewer than n_splits patients, sklearn's
        # StratifiedKFold raises. We collapse rare subtypes into
        # ``"__rare__"`` so stratification still runs (and the rare-subtype
        # mix is preserved across folds in aggregate even if not per-fold).
        from collections import Counter

        counts = Counter(labels)
        rare = {label for label, n in counts.items() if n < self.n_splits}
        if rare:
            labels = [label if label not in rare else "__rare__" for label in labels]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(ids, labels)):
            yield SplitFold(
                fold_id=f"{self.name}__fold{fold_idx}",
                split_name=self.name,
                train_patient_ids=tuple(ids[i] for i in train_idx),
                test_patient_ids=tuple(ids[i] for i in test_idx),
                seed=self.seed,
                metadata={"fold_index": str(fold_idx), "n_splits": str(self.n_splits)},
            )
