"""Loaders that turn raw dataset artifacts into validated tranches.

``load_tranche`` is the entry point the analysis scripts use: it serves the two
MVP cohorts (Soragni ``sarcoma``, GDSC2 ``gdscv2``) from our own raw-artifact
loaders and falls back to CoderData for anything else. The native bundles are
adapted to the shared ``CoderDataBundle`` shape that ``build_sample_design``
consumes.
"""

from fmharness.data.loaders.adapt import (
    adapt_gdsc2,
    adapt_soragni,
    load_tranche,
)
from fmharness.data.loaders.coderdata import (
    CoderDataBundle,
    IngestError,
    load_coderdata_tranche,
)
from fmharness.data.loaders.gdsc2_sarcoma import (
    GDSC2SarcomaBundle,
    load_gdsc2_sarcoma,
)
from fmharness.data.loaders.soragni import (
    SoragniBundle,
    canonicalize_patient_id,
    load_soragni,
)

__all__ = [
    "CoderDataBundle",
    "GDSC2SarcomaBundle",
    "IngestError",
    "SoragniBundle",
    "adapt_gdsc2",
    "adapt_soragni",
    "canonicalize_patient_id",
    "load_coderdata_tranche",
    "load_gdsc2_sarcoma",
    "load_soragni",
    "load_tranche",
]
