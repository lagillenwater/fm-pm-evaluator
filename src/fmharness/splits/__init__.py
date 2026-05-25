"""OOD splits for the harness.

Three splitters share a common ``Splitter`` Protocol + ``SplitFold`` shape:

- ``StratifiedInDistribution`` -- K-fold stratified by subtype (baseline)
- ``LeavePatientOut`` -- one fold per patient (OOD by individual)
- ``LeaveSubtypeOut`` -- one fold per subtype (OOD by histology, configurable
  fine vs coarse granularity via an optional ``subtype_map``)

Downstream callers go through ``require_split(splitter)`` so the harness
refuses to silently run unsplit predictions.
"""

from fmharness.splits.base import (
    MissingSplitError,
    SplitFold,
    SplittablePatient,
    Splitter,
    require_split,
)
from fmharness.splits.lpo import LeavePatientOut
from fmharness.splits.lso import LeaveSubtypeOut, LSOGranularity
from fmharness.splits.stratified import StratifiedInDistribution

__all__ = [
    "LSOGranularity",
    "LeavePatientOut",
    "LeaveSubtypeOut",
    "MissingSplitError",
    "SplitFold",
    "SplittablePatient",
    "Splitter",
    "StratifiedInDistribution",
    "require_split",
]
