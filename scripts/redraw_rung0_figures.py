"""Redraw rung 0's split and permutation figures from committed tables, and re-record checksums.

Both figures are functions of tables the run committed: the split figure of the pool description
and the per-condition table, the permutation figures of each gene set's different-drug permutation
means, the pooled null draws, and the summary rows. When a figure's drawing code is corrected after
the run -- as on 2026-09-11, when the re-audit found the split panel counting unreplicated triples
and the permutation figures mixing strata -- redrawing from those tables gives the figure the run
would now draw, without repeating a scan or a permutation. The checksum record is rewritten after,
so the battery checks the figures a reader sees.

    uv run python scripts/redraw_rung0_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import delta_reproducibility as dr  # noqa: E402

from fmharness import figures as fg  # noqa: E402


def read(path: Path) -> pd.DataFrame:
    """A committed table, keeping the cell line whose DepMap identifier is the literal NA."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", type=Path, default=REPO / "docs/tasks/rung0-assay-reliability")
    task = ap.parse_args().task_dir
    figures = task / "figures"
    fg.fig_split(
        read(task / "rung0_pool_description.csv"),
        read(task / "rung0_per_pair_r.csv"),
        figures / "02_split.png",
    )
    print(f"redrew {figures / '02_split.png'}")
    reliability = read(task / "rung0_reliability.csv").iloc[0].to_dict()
    draws = read(task / "rung0_null_draws.csv")
    for gene_set, suffix in (("all", ""), ("responder", "_responder")):
        summary = task / f"rung0_permutation_summary{suffix}.csv"
        means = task / f"rung0_permutation_perm_means_diff_drug{suffix}.csv"
        if not (summary.exists() and means.exists()):
            print(f"no permutation outputs for {gene_set}: figure not drawn")
            continue
        out = figures / f"11_permutation_vs_bootstrap{suffix}.png"
        fg.fig_permutation_vs_bootstrap(
            read(means),
            draws,
            {**reliability, **read(summary).iloc[0].to_dict()},
            out,
            gene_set=gene_set,
        )
        print(f"redrew {out}")
    dr.write_audit_checksums(task)


if __name__ == "__main__":
    main()
