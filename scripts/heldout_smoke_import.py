"""Smoke-import every rung 1 module, without running any of its ``main()``.

Alpine's ``stack`` environment is pinned to Python 3.10 (global-constraints.md), while rung 1
is developed under a newer interpreter locally. This script is the check task 11's jobs run
first, before any long stage: import every ``fmharness.heldout`` module and every
``scripts/heldout_*.py`` module, under whichever interpreter is asked to run it, and fail loudly
(non-zero exit, via an uncaught exception) the moment a 3.11-only construct or a missing
dependency would otherwise surface hours into a cluster job instead.

    uv run python scripts/heldout_smoke_import.py
    uv run --python 3.10 --isolated --no-project --with numpy --with pandas --with scipy \\
        --with scikit-learn --with pyarrow --with duckdb --with pydantic --with rdkit \\
        --with anndata --with matplotlib env PYTHONPATH=src python scripts/heldout_smoke_import.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

HELDOUT_MODULES = (
    "fmharness.heldout",
    "fmharness.heldout.answers",
    "fmharness.heldout.cells",
    "fmharness.heldout.chemistry",
    "fmharness.heldout.comparisons",
    "fmharness.heldout.controls",
    "fmharness.heldout.descriptions",
    "fmharness.heldout.figures",
    "fmharness.heldout.grid",
    "fmharness.heldout.leakage",
    "fmharness.heldout.models",
    "fmharness.heldout.records",
    "fmharness.heldout.scoring",
)


def _script_paths() -> list[Path]:
    """Every ``scripts/heldout_*.py`` file, in a stable order."""
    return sorted((REPO / "scripts").glob("heldout_*.py"))


def _import_by_path(path: Path) -> None:
    """Import ``path`` as a module named after its stem, without executing its ``main()``.

    Registers it in ``sys.modules`` first, as the project's own tests do, so a module that
    imports a sibling script by plain name (``import delta_reproducibility``) resolves it
    through ``sys.path`` rather than failing to find it.
    """
    name = path.stem
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main() -> None:
    for name in HELDOUT_MODULES:
        importlib.import_module(name)
    for path in _script_paths():
        _import_by_path(path)
    print(f"python {sys.version.split()[0]}")
    print("ok")


if __name__ == "__main__":
    main()
