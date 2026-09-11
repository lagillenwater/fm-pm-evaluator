"""Task 3 of rung 1: scan Tahoe's pseudobulk DE table into the grid's answer, and the line
crosswalk task 4 needs.

Reuses rung 0's DuckDB helpers (``scripts/delta_reproducibility.py``) to scan the 89 GB table
in gene slices, on the cluster. Three modes:

* One gene slice (``--part k --n-parts N``): keeps rows with a plate, a grid drug, and a grid
  line, at 5 uM, groups by (line, drug, gene), and writes ``answers_{k}.parquet`` (columns
  ``line, drug, gene, mean_lfc, min_padj, n_lfc, n_plates``) into ``--cache``.
* ``--combine``: refuses (non-zero exit) unless every one of the ``--n-parts`` slices is
  done, then assembles the grid's dense answer arrays and writes ``answers.npz`` into
  ``--cache`` and ``rung1_answer_counts.csv`` into ``--out-dir``.
* ``--crosswalk``: the distinct (line, cellosaurus, cell_name) triples for the grid's lines,
  read from the key columns alone, written to ``rung1_line_crosswalk.csv`` in ``--out-dir``.

Every output is followed by a ``<name>.done.json`` completion record; a rerun skips an output
whose record is already valid.

    uv run python scripts/heldout_answers.py --local-dir /scratch/.../tahoe_pseudobulk_de \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --cache /scratch/.../rung1_cache --part 0 --n-parts 64
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import delta_reproducibility as dr  # noqa: E402

from fmharness.heldout.answers import assemble_answers  # noqa: E402
from fmharness.heldout.grid import DOSE_UM, Grid, load_grid  # noqa: E402
from fmharness.heldout.records import is_done, write_record  # noqa: E402


def _de_paths(local_dir: Path) -> list[str]:
    """Every ``pseudobulk_differential_expression`` parquet file under ``local_dir`` (rung 0's
    own way of finding the table's files)."""
    paths = sorted(
        str(p)
        for p in local_dir.rglob("*.parquet")
        if "pseudobulk_differential_expression" in str(p)
    )
    if not paths:
        raise SystemExit(
            f"no pseudobulk_differential_expression parquet files found under {local_dir}"
        )
    return paths


def slice_answers(
    paths: list[str],
    drugs: tuple[str, ...],
    lines: tuple[str, ...],
    dose: float,
    tmp: Path,
    *,
    n_parts: int,
    part: int,
    memory_limit: str = "20GB",
    threads: int | None = None,
) -> pd.DataFrame:
    """One gene slice of the grid's answer rows: (line, drug, gene) x its answer aggregates.

    Keeps rows with a plate, a grid drug, and a grid line, at ``dose`` (matched after rounding
    both sides to 6 decimals, since the table's dose is float32). Groups by (line, drug, gene):
    the mean ``log2FoldChange`` over the pair's plates (DuckDB's ``avg`` ignores nulls, so one
    untestable plate gives the other's value), the minimum adjusted p-value, the count of
    non-null fold changes, and the count of distinct plates. Columns come back named
    ``line, drug, gene, mean_lfc, min_padj, n_lfc, n_plates``.
    """
    con = dr._connect(tmp, memory_limit, threads)
    _cols, repl, dose_col = dr._pool_columns(con, paths, None)
    where_drug, drug_params = dr._drug_predicate(list(drugs), alias="t.")
    gene_part = dr._gene_partition(n_parts, part, alias="t.")
    query = f"""
        SELECT t.Cell_ID_DepMap AS patient,
               t.drug AS drug,
               t.gene_name AS gene_name,
               avg(t.log2FoldChange) AS mean_lfc,
               min(t.padj) AS min_padj,
               count(t.log2FoldChange) AS n_lfc,
               count(DISTINCT t.{repl}) AS n_plates
        FROM read_parquet(?) t
        WHERE t.{repl} IS NOT NULL
          AND round(CAST(t.{dose_col} AS DOUBLE), 6) = ?
          AND t.Cell_ID_DepMap IN (SELECT unnest(?)){where_drug}{gene_part}
        GROUP BY 1, 2, 3
    """
    result = con.execute(query, [paths, round(float(dose), 6), list(lines), *drug_params])
    df = dr._compact_df(result)
    return df.rename(columns={"patient": "line", "gene_name": "gene"})


def crosswalk_rows(
    paths: list[str],
    lines: tuple[str, ...],
    tmp: Path,
    memory_limit: str,
    threads: int | None,
) -> pd.DataFrame:
    """The distinct (line, cellosaurus, cell_name) triples for ``lines``, key columns only."""
    con = dr._connect(tmp, memory_limit, threads)
    result = con.execute(
        """SELECT DISTINCT Cell_ID_DepMap AS line,
                  Cell_ID_Cellosaur AS cellosaurus,
                  Cell_Name_Vevo AS cell_name
           FROM read_parquet(?)
           WHERE Cell_ID_DepMap IN (SELECT unnest(?))""",
        [paths, list(lines)],
    )
    return result.df()


def build_crosswalk(rows: pd.DataFrame, lines: tuple[str, ...]) -> pd.DataFrame:
    """One row per grid line, in grid order: its Cellosaurus id and Vevo cell name.

    Raises ``ValueError`` if a grid line is missing from ``rows``, or maps to more than one
    Cellosaurus id -- the crosswalk task 4 needs would then be ambiguous.
    """
    counts = rows.groupby("line")["cellosaurus"].nunique()
    present = set(counts.index)
    missing = [line for line in lines if line not in present]
    if missing:
        raise ValueError(f"grid lines missing from the crosswalk source: {missing}")
    multi = sorted(line for line in lines if counts.loc[line] > 1)
    if multi:
        raise ValueError(f"lines with more than one Cellosaurus id: {multi}")
    return (
        rows.drop_duplicates(subset=["line"])
        .set_index("line")
        .loc[list(lines)]
        .reset_index()[["line", "cellosaurus", "cell_name"]]
    )


def combine_answers(cache: Path, grid: Grid, n_parts: int, out_dir: Path) -> None:
    """Assemble every gene slice into ``answers.npz`` and ``rung1_answer_counts.csv``.

    Refuses (raises ``SystemExit``, naming the missing parts) unless every one of the
    ``n_parts`` slices has a valid completion record.
    """
    parts = [cache / f"answers_{k}.parquet" for k in range(n_parts)]
    missing = [str(p) for p in parts if not is_done(p)]
    if missing:
        raise SystemExit(
            f"combine refuses: {len(missing)} of {n_parts} answer slices are missing or "
            f"invalid, run --part for each of: {missing}"
        )

    rows = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    answers = assemble_answers(rows, grid)

    npz_path = cache / "answers.npz"
    np.savez(
        npz_path,
        lines=np.array(answers.lines, dtype=str),
        drugs=np.array(answers.drugs, dtype=str),
        genes=np.array(answers.genes, dtype=str),
        delta=answers.delta,
        responding=answers.responding,
    )
    write_record(npz_path)
    print(f"wrote {npz_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = out_dir / "rung1_answer_counts.csv"
    n_lines, n_drugs = len(answers.lines), len(answers.drugs)
    n_tested = np.isfinite(answers.delta).sum(axis=2)
    n_responding = answers.responding.sum(axis=2)
    counts = pd.DataFrame(
        {
            "line": np.repeat(np.array(answers.lines, dtype=object), n_drugs),
            "drug": np.tile(np.array(answers.drugs, dtype=object), n_lines),
            "n_tested": n_tested.reshape(-1),
            "n_responding": n_responding.reshape(-1),
        }
    )
    counts.to_csv(counts_path, index=False)
    write_record(counts_path)
    print(f"wrote {counts_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--local-dir", type=Path)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--part", type=int)
    ap.add_argument("--n-parts", type=int)
    ap.add_argument("--memory-limit", default="20GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--crosswalk", action="store_true")
    args = ap.parse_args()

    grid = load_grid(args.grid)

    if args.crosswalk:
        if args.local_dir is None or args.out_dir is None:
            raise SystemExit("--crosswalk needs --local-dir and --out-dir")
        paths = _de_paths(args.local_dir)
        tmp = dr._private_tmp(args.local_dir, "crosswalk")
        try:
            rows = crosswalk_rows(paths, grid.lines, tmp, args.memory_limit, args.threads)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        table = build_crosswalk(rows, grid.lines)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "rung1_line_crosswalk.csv"
        table.to_csv(out_path, index=False)
        write_record(out_path)
        print(f"wrote {out_path}")
        return

    if args.combine:
        if args.cache is None:
            raise SystemExit("--combine needs --cache")
        if args.n_parts is None:
            raise SystemExit("--combine needs --n-parts")
        out_dir = args.out_dir if args.out_dir is not None else args.cache
        combine_answers(args.cache, grid, args.n_parts, out_dir)
        return

    if args.local_dir is None or args.cache is None or args.part is None or args.n_parts is None:
        raise SystemExit(
            "need --local-dir --cache --part --n-parts for a slice "
            "(or --combine, or --crosswalk --local-dir --out-dir)"
        )
    out_path = args.cache / f"answers_{args.part}.parquet"
    if is_done(out_path):
        print(f"{out_path} already done, skipping")
        return

    paths = _de_paths(args.local_dir)
    tmp = dr._private_tmp(args.local_dir, f"answers_{args.part}")
    try:
        rows = slice_answers(
            paths,
            grid.drugs,
            grid.lines,
            DOSE_UM,
            tmp,
            n_parts=args.n_parts,
            part=args.part,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    args.cache.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_path, index=False)
    write_record(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
