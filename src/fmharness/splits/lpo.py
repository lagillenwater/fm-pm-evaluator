"""Leave-patient-out CV split: N folds for N patients.

Each fold holds one patient out as test; the other N-1 form the train set.
The seed is recorded but has no effect on fold composition (LPO is fully
deterministic given the patient set) -- it exists to flow into the
``EnvironmentSnapshot`` of downstream prediction records.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from fmharness.splits.base import SplitFold, SplittablePatient


class LeavePatientOut:
    """One fold per patient. Patient_id is the held-out unit."""

    name = "leave_patient_out"

    def __init__(self, *, seed: int) -> None:
        self.seed = seed

    def split(self, patients: Sequence[SplittablePatient]) -> Iterator[SplitFold]:
        if len(patients) < 2:
            raise ValueError(f"need >= 2 patients for LPO; got {len(patients)}")
        ids = [p.patient_id for p in patients]
        for held_out in ids:
            train = tuple(i for i in ids if i != held_out)
            yield SplitFold(
                fold_id=f"{self.name}__{held_out}",
                split_name=self.name,
                train_patient_ids=train,
                test_patient_ids=(held_out,),
                seed=self.seed,
                metadata={"held_out_patient": held_out},
            )
