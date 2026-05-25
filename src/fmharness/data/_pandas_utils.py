"""Internal helpers for coercing pandas cell values to plain Python types.

Loaders read DataFrame rows and need to turn nullable cells (``None``,
``pd.NA``, ``NaN``) into ``None`` and concrete cells into ``int`` / ``str``.
The natural ``int(v) if not pd.isna(v) else None`` does not narrow under
pyright (``pd.isna`` returns ``bool | NDArray | NDFrame`` for ``object``
input), so these helpers do the narrowing explicitly with ``isinstance``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def maybe_int(v: object) -> int | None:
    """Coerce a pandas cell to ``int``, or ``None`` if missing.

    Handles ``None``, ``pd.NA``, ``NaN`` floats, Python ``int`` / ``float``,
    numpy scalar types (``np.int64``, ``np.float64``, ...), and numeric strings.
    """
    if v is None or v is pd.NA:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if math.isnan(float(v)) else int(v)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) else int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def maybe_str(v: object) -> str | None:
    """Coerce a pandas cell to ``str``, or ``None`` if missing."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, np.floating) and math.isnan(float(v)):
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return str(v)
