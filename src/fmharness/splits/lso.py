"""Leave-subtype-out CV split: one fold per unique subtype label.

Configurable granularity:

- ``"fine"`` (default): use ``Patient.subtype`` verbatim. For the GDSC2
  sarcoma cohort this is the OncotreeSubtype (Ewing Sarcoma, Osteosarcoma,
  Chondrosarcoma, ...); for Soragni it is the drug-screen Diagnosis
  free-text label.
- ``"coarse"``: collapse fine labels to broader families via a ``subtype_map``
  passed by the caller (e.g. ``{"Dedifferentiated Chondrosarcoma":
  "Chondrosarcoma", "Embryonal Rhabdomyosarcoma": "Rhabdomyosarcoma"}``).
  Unmapped labels fall through unchanged.

Patients with no subtype (``Patient.subtype is None``) are grouped under
``"__missing__"`` and form their own fold. The caller can filter these
upstream if that fold isn't useful.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from fmharness.splits.base import SplitFold, SplittablePatient

_MISSING_SUBTYPE = "__missing__"
LSOGranularity = Literal["fine", "coarse"]


class LeaveSubtypeOut:
    """One fold per unique (mapped) subtype. Subtype is the held-out unit."""

    name = "leave_subtype_out"

    def __init__(
        self,
        *,
        seed: int,
        granularity: LSOGranularity = "fine",
        subtype_map: dict[str, str] | None = None,
    ) -> None:
        if granularity == "coarse" and not subtype_map:
            raise ValueError(
                "granularity='coarse' requires a non-empty subtype_map; "
                "pass {fine_label: coarse_label, ...}"
            )
        self.seed = seed
        self.granularity = granularity
        self.subtype_map = subtype_map or {}

    def _map_label(self, subtype: str | None) -> str:
        if subtype is None:
            return _MISSING_SUBTYPE
        if self.granularity == "fine":
            return subtype
        return self.subtype_map.get(subtype, subtype)

    def split(self, patients: list[SplittablePatient]) -> Iterator[SplitFold]:
        if len(patients) < 2:
            raise ValueError(f"need >= 2 patients for LSO; got {len(patients)}")
        # Group patient_ids by their (mapped) subtype
        groups: dict[str, list[str]] = {}
        for p in patients:
            key = self._map_label(p.subtype)
            groups.setdefault(key, []).append(p.patient_id)
        if len(groups) < 2:
            raise ValueError(
                f"LSO needs >= 2 distinct subtypes; got {len(groups)}: {sorted(groups)}"
            )

        for held_out_subtype, test_ids in sorted(groups.items()):
            train = tuple(
                pid for label, ids in groups.items() for pid in ids if label != held_out_subtype
            )
            yield SplitFold(
                fold_id=f"{self.name}__{held_out_subtype}",
                split_name=self.name,
                train_patient_ids=train,
                test_patient_ids=tuple(test_ids),
                seed=self.seed,
                metadata={
                    "held_out_subtype": held_out_subtype,
                    "granularity": self.granularity,
                    "test_size": str(len(test_ids)),
                },
            )
