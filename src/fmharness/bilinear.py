"""Bilinear feature construction shared by the structure-aware transfer models.

The drug-structure-aware models predict ``AUC(s, d) = ridge([z_s, g_d, z_s (x) g_d])``,
where ``z_s`` is the sample (expression) representation, ``g_d`` the drug
(fingerprint) representation, and ``z_s (x) g_d`` their outer product -- the
drug-specific interaction term mediated by chemistry. Both the screen-free
transfer (``scripts/transfer_pharmaformer_lite.py``) and the per-patient eval
(``scripts/per_patient_eval.py``) build the same design block, so it lives here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bilinear_features(z: NDArray[np.float64], g: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stack ``[z, g, z (x) g]`` row-wise (outer product flattened per row)."""
    z = np.asarray(z, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    inter = np.einsum("ij,ik->ijk", z, g).reshape(len(z), -1)
    return np.hstack([z, g, inter])
