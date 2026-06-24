"""Shared plotting setup for the figure scripts.

Selects the headless Agg backend (no display server on HPC), re-exports ``plt``,
and gives one ``savefig`` that every figure script uses so the save idiom (make
the parent dir, dpi 150, tight bbox, print the path) lives in one place.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["plt", "savefig"]


def savefig(fig: plt.Figure, path: str | Path) -> Path:
    """Write ``fig`` to ``path`` (mkdir parents, dpi=150, tight bbox) and print it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return out
