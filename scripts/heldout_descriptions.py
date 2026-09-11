"""Build rung 1's non-Stack line descriptions, drug fingerprints, and the build known-answer
control (design.md sections 4, 6, 8; the brief's task 5).

Reads task 4's cache (``expression.parquet``, ``expression_halves.parquet``, ``cells/line_
{i}.h5ad``) and the grid (``rung1_grid.json``) and drug metadata table, and writes:

* ``descriptions.npz`` (cache) -- expression (standardized), PCA scores and loadings, NMF W
  for each k in (2, 5, 10, 15, 20) (each standardized), and a random stand-in per description.
* ``tanimoto.npz`` (cache) -- Morgan fingerprints and Tanimoto similarity of the grid's 107
  drugs, in grid drug order.
* ``rung1_identity_match.csv`` (out-dir) -- half-vs-half identity match for expression, PCA
  (20 components) and NMF (k = 20), beside its shuffled null's mean and 99th percentile.
* ``rung1_identity_grid_{description}.csv`` (out-dir) -- the long-form line x line correlation
  grid behind that figure, real beside the first shuffle.

Each output is followed by a ``<name>.done.json`` completion record; a rerun skips outputs
already done and valid.

    uv run python scripts/heldout_descriptions.py \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --drug-metadata path/to/drug_metadata.parquet \\
        --cache /scratch/alpine/$USER/rung1_cache \\
        --out-dir docs/tasks/rung1-held-out-prediction \\
        --n-shuffles 200 --seed 0
"""

# pandas, scipy and anndata ship no PEP-561 type stubs in this environment; under strict mode
# that turns every call site into a cascade of reportUnknown* noise about *their* types, not
# ours. Same suppression, same rationale as the rest of this project's pyright strict config
# where it touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fmharness.heldout.chemistry import morgan_fingerprints, tanimoto  # noqa: E402
from fmharness.heldout.descriptions import (  # noqa: E402
    NMF_SEED,
    RANDOM_SEEDS,
    identity_correlations,
    identity_match,
    nmf_components,
    nmf_project,
    pca_components,
    pca_project,
    random_stand_in,
    shuffled_identity_correlations,
    shuffled_identity_null,
    standardize_apply,
    standardize_stats,
)
from fmharness.heldout.grid import Grid, load_drug_metadata, load_grid  # noqa: E402
from fmharness.heldout.records import is_done, write_record  # noqa: E402

NMF_KS: tuple[int, ...] = (2, 5, 10, 15, 20)


def _expression_full(cache: Path, lines: tuple[str, ...]) -> pd.DataFrame:
    """``expression.parquet``, reindexed to grid line order; raises if any grid line is
    missing (never silently drops a line or leaves rows in the wrong order)."""
    expr = pd.read_parquet(cache / "expression.parquet")
    missing = sorted(set(lines) - set(expr.index.astype(str)))
    if missing:
        raise ValueError(f"expression.parquet is missing grid lines: {missing}")
    return cast(pd.DataFrame, expr.loc[list(lines)])


def _expression_halves(cache: Path, lines: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``expression_halves.parquet``'s two halves, each reindexed to grid line order."""
    halves = pd.read_parquet(cache / "expression_halves.parquet")
    half0 = halves.xs(0, level="half")
    half1 = halves.xs(1, level="half")
    for name, half in (("half 0", half0), ("half 1", half1)):
        missing = sorted(set(lines) - set(half.index.astype(str)))
        if missing:
            raise ValueError(f"expression_halves.parquet {name} is missing grid lines: {missing}")
    return half0.loc[list(lines)], half1.loc[list(lines)]


def _load_all_cells(
    cache: Path, lines: tuple[str, ...]
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Every grid line's ``cells/line_{i}.h5ad`` (task 4's cache), concatenated into one raw
    counts matrix with matching ``line`` and ``half`` arrays -- the population
    ``shuffled_identity_null`` shuffles cells' line labels over."""
    import anndata as ad

    counts: list[sparse.csr_matrix] = []
    line_labels: list[np.ndarray] = []
    half_labels: list[np.ndarray] = []
    for i, line in enumerate(lines):
        path = cache / "cells" / f"line_{i}.h5ad"
        adata = ad.read_h5ad(path)
        x = cast(sparse.csr_matrix, adata.X)
        counts.append(x.tocsr())
        obs_line = adata.obs["line"].to_numpy()
        assert (obs_line == line).all(), f"{path} obs['line'] does not match grid line {line!r}"
        line_labels.append(obs_line)
        half_labels.append(adata.obs["half"].to_numpy(dtype=np.int64))
    return (
        cast(sparse.csr_matrix, sparse.vstack(counts, format="csr")),
        np.concatenate(line_labels),
        np.concatenate(half_labels),
    )


def build_descriptions(
    grid: Grid, cache: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Every description matrix (for ``descriptions.npz``) and the extra full-data fitted
    parameters (means, SDs, loadings, NMF ``H``) the identity-match step reuses to describe a
    half consistently with how the full data was described."""
    lines = grid.lines
    n_lines = len(lines)

    expr_full = _expression_full(cache, lines)
    genes = np.asarray(expr_full.columns.astype(str))
    e = expr_full.to_numpy(dtype=np.float64)
    mean_e, std_e = standardize_stats(e)
    expression = standardize_apply(e, mean_e, std_e)

    pca_scores, pca_loadings = pca_components(expression, k_max=20)

    nmf_arrays: dict[str, np.ndarray] = {}
    nmf_h_by_k: dict[int, np.ndarray] = {}
    nmf_stats_by_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in NMF_KS:
        w, h = nmf_components(e, k, seed=NMF_SEED)
        mean_w, std_w = standardize_stats(w)
        nmf_arrays[f"nmf_{k}"] = standardize_apply(w, mean_w, std_w)
        nmf_h_by_k[k] = h
        nmf_stats_by_k[k] = (mean_w, std_w)

    descriptions: dict[str, np.ndarray] = {
        "lines": np.asarray(lines),
        "expression": expression,
        "expression_genes": genes,
        "pca": pca_scores,
        "pca_loadings": pca_loadings,
        **nmf_arrays,
        "random_expression": random_stand_in(n_lines, e.shape[1], RANDOM_SEEDS["expression"]),
        "random_pca": random_stand_in(n_lines, 20, RANDOM_SEEDS["pca"]),
        "random_nmf": random_stand_in(n_lines, 20, RANDOM_SEEDS["nmf"]),
    }

    fitted: dict[str, np.ndarray] = {
        "mean_e": mean_e,
        "std_e": std_e,
        "pca_loadings": pca_loadings,
        "nmf_h_20": nmf_h_by_k[20],
        "nmf_mean_w_20": nmf_stats_by_k[20][0],
        "nmf_std_w_20": nmf_stats_by_k[20][1],
    }
    return descriptions, fitted


def build_tanimoto(grid: Grid, drug_metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Morgan fingerprints and Tanimoto similarity of the grid's drugs, in grid drug order."""
    smiles_by_name = dict(
        zip(
            drug_metadata["drug"].astype(str),
            drug_metadata["canonical_smiles"].astype(str),
            strict=True,
        )
    )
    smiles = [smiles_by_name[grid.metadata_name[drug]] for drug in grid.drugs]
    fingerprints = morgan_fingerprints(smiles)
    similarity = tanimoto(fingerprints)
    return {
        "drugs": np.asarray(grid.drugs),
        "similarity": similarity,
        "fingerprints": fingerprints,
    }


def identity_match_tables(
    grid: Grid,
    cache: Path,
    fitted: dict[str, np.ndarray],
    n_shuffles: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``rung1_identity_match.csv``'s rows and each description's long-form correlation grid,
    for expression, PCA (20 components) and NMF (k = 20)."""
    lines = grid.lines
    n_lines = len(lines)
    half0, half1 = _expression_halves(cache, lines)
    half0_raw = half0.to_numpy(dtype=np.float64)
    half1_raw = half1.to_numpy(dtype=np.float64)

    mean_e = fitted["mean_e"]
    std_e = fitted["std_e"]
    pca_loadings = fitted["pca_loadings"]
    nmf_h_20 = fitted["nmf_h_20"]
    nmf_mean_w_20 = fitted["nmf_mean_w_20"]
    nmf_std_w_20 = fitted["nmf_std_w_20"]

    def describe_expression(x: np.ndarray) -> np.ndarray:
        return x

    def describe_pca(x: np.ndarray) -> np.ndarray:
        return pca_project(x, mean_e, std_e, pca_loadings)

    def describe_nmf(x: np.ndarray) -> np.ndarray:
        w = nmf_project(x, nmf_h_20, seed=NMF_SEED)
        return standardize_apply(w, nmf_mean_w_20, nmf_std_w_20)

    counts, cell_lines, cell_halves = _load_all_cells(cache, lines)

    rows: list[dict[str, float | str | int]] = []
    grids: dict[str, pd.DataFrame] = {}
    for name, describe in (
        ("expression", describe_expression),
        ("pca", describe_pca),
        ("nmf", describe_nmf),
    ):
        a = describe(half0_raw)
        b = describe(half1_raw)
        real_share = identity_match(a, b)
        null = shuffled_identity_null(counts, cell_lines, cell_halves, describe, n_shuffles, seed)
        rows.append(
            {
                "description": name,
                "identity_share": real_share,
                "null_mean": float(null.mean()),
                "null_p99": float(np.quantile(null, 0.99)),
                "n_lines": n_lines,
                "n_shuffles": n_shuffles,
            }
        )

        real_grid = identity_correlations(a, b)
        real_long = pd.DataFrame(
            {
                "line_a": np.repeat(lines, n_lines),
                "line_b": np.tile(lines, n_lines),
                "r": real_grid.ravel(),
                "source": "real",
            }
        )

        shuffled_categories, shuffled_corr = shuffled_identity_correlations(
            counts, cell_lines, cell_halves, describe, seed
        )
        n_shuffled = len(shuffled_categories)
        shuffled_long = pd.DataFrame(
            {
                "line_a": np.repeat(shuffled_categories, n_shuffled),
                "line_b": np.tile(shuffled_categories, n_shuffled),
                "r": shuffled_corr.ravel(),
                "source": "shuffled",
            }
        )
        grids[name] = pd.concat([real_long, shuffled_long], ignore_index=True)

    return pd.DataFrame(rows), grids


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--drug-metadata", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    descriptions_path = args.cache / "descriptions.npz"
    tanimoto_path = args.cache / "tanimoto.npz"
    identity_match_path = args.out_dir / "rung1_identity_match.csv"

    grid = load_grid(args.grid)

    need_descriptions = not is_done(descriptions_path)
    need_identity_match = not is_done(identity_match_path)
    # ``fitted`` (the full-data mean/SD/loadings/NMF H the identity-match step reuses to
    # describe a half consistently with the full data) is cheap only relative to re-scanning
    # the cache; it is recomputed whenever either downstream output still needs building,
    # rather than trusting a previous run's in-memory state that no longer exists.
    descriptions: dict[str, np.ndarray] | None = None
    fitted: dict[str, np.ndarray] | None = None
    if need_descriptions or need_identity_match:
        descriptions, fitted = build_descriptions(grid, args.cache)

    if need_descriptions:
        assert descriptions is not None
        np.savez(descriptions_path, **cast(dict[str, Any], descriptions))
        write_record(descriptions_path)
        print(f"wrote {descriptions_path}")
    else:
        print(f"{descriptions_path} already done, skipping")

    if is_done(tanimoto_path):
        print(f"{tanimoto_path} already done, skipping")
    else:
        drug_metadata = load_drug_metadata(args.drug_metadata)
        tanimoto_arrays = build_tanimoto(grid, drug_metadata)
        np.savez(tanimoto_path, **cast(dict[str, Any], tanimoto_arrays))
        write_record(tanimoto_path)
        print(f"wrote {tanimoto_path}")

    if not need_identity_match:
        print(f"{identity_match_path} already done, skipping")
        return

    assert fitted is not None
    identity_match_table, grids = identity_match_tables(
        grid, args.cache, fitted, args.n_shuffles, args.seed
    )
    identity_match_table.to_csv(identity_match_path, index=False)
    write_record(identity_match_path)
    print(f"wrote {identity_match_path}")

    for name, table in grids.items():
        grid_path = args.out_dir / f"rung1_identity_grid_{name}.csv"
        table.to_csv(grid_path, index=False)
        write_record(grid_path)
        print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
