"""Task 5 of rung 1: line descriptions and chemistry.

Design.md section 4 fixes what non-Stack line descriptions are (expression, PCA, NMF, a random
stand-in) and how a hidden drug is described (a Morgan fingerprint, compared by Tanimoto
similarity); section 8 fixes the build step's known-answer control -- a description built from
half a line's cells should best match that line's other half, and fall to chance once cells'
line labels are shuffled first. Every test below runs the real, shipped functions.
"""

# pandas, scipy and rdkit ship no PEP-561 type stubs in this environment; under strict mode
# that turns every call site into a cascade of reportUnknown* noise about *their* types, not
# ours. Same suppression, same rationale as the rest of this project's pyright strict config
# where it touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from scipy import sparse

from fmharness.heldout.cells import pseudobulk_log_cpm
from fmharness.heldout.chemistry import morgan_fingerprints, tanimoto
from fmharness.heldout.descriptions import (
    RANDOM_SEEDS,
    identity_match,
    linear_kernel,
    nmf_components,
    nmf_project,
    pca_components,
    pca_project,
    random_stand_in,
    shuffled_identity_null,
    standardize,
    standardize_apply,
    standardize_stats,
)
from fmharness.heldout.grid import attach_drug_metadata, grid_from_rung0, load_drug_metadata
from fmharness.heldout.records import is_done

REPO = Path(__file__).resolve().parents[1]
PER_PAIR_PATH = REPO / "results/rung0-assay-reliability/rung0_per_pair_r.csv"
DRUG_METADATA_PATH = REPO / "tests/fixtures/tahoe_drug_metadata.csv"

pytestmark = pytest.mark.known_answer


# ==============================================================================================
# standardize
# ==============================================================================================


def test_standardize_zero_mean_unit_sd_and_zero_variance_column_becomes_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(20, 4))
    x[:, 2] = 7.0  # a constant column: zero variance across lines
    out = standardize(x)
    np.testing.assert_allclose(out.mean(axis=0)[[0, 1, 3]], 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0, ddof=0)[[0, 1, 3]], 1.0, atol=1e-10)
    np.testing.assert_array_equal(out[:, 2], 0.0)


def test_standardize_apply_uses_given_stats_not_its_own() -> None:
    rng = np.random.default_rng(1)
    full = rng.normal(loc=2.0, scale=4.0, size=(30, 3))
    mean, std = standardize_stats(full)
    half = full[:10]
    applied = standardize_apply(half, mean, std)
    expected = (half - mean) / std
    np.testing.assert_allclose(applied, expected)


# ==============================================================================================
# pca_components
# ==============================================================================================


def test_pca_matches_svd_up_to_fixed_sign_and_the_sign_rule_holds() -> None:
    rng = np.random.default_rng(2)
    x_std = standardize(rng.normal(size=(15, 8)))
    scores, loadings = pca_components(x_std, k_max=5)

    u, s, vt = np.linalg.svd(x_std, full_matrices=False)
    raw_scores = u[:, :5] * s[:5]
    raw_loadings = vt[:5]

    # Recover the per-component sign flip and check scores/loadings only differ by it.
    largest_idx = np.argmax(np.abs(raw_loadings), axis=1)
    sign = np.sign(raw_loadings[np.arange(5), largest_idx])
    sign[sign == 0] = 1.0
    np.testing.assert_allclose(loadings, raw_loadings * sign[:, None], atol=1e-8)
    np.testing.assert_allclose(scores, raw_scores * sign[None, :], atol=1e-8)

    # The sign rule: each component's largest-magnitude loading entry is positive.
    for c in range(loadings.shape[0]):
        idx = np.argmax(np.abs(loadings[c]))
        assert loadings[c, idx] > 0


def test_pca_project_half_reproduces_full_projection_when_half_equals_full() -> None:
    rng = np.random.default_rng(3)
    e = np.abs(rng.normal(loc=5.0, size=(10, 6)))
    mean, std = standardize_stats(e)
    e_std = standardize_apply(e, mean, std)
    scores, loadings = pca_components(e_std, k_max=4)
    projected = pca_project(e, mean, std, loadings)
    np.testing.assert_allclose(projected, scores, atol=1e-8)


# ==============================================================================================
# nmf_components / nmf_project
# ==============================================================================================


def test_nmf_reproducible_under_a_seed() -> None:
    rng = np.random.default_rng(4)
    x = np.abs(rng.normal(loc=3.0, size=(12, 9)))
    w1, h1 = nmf_components(x, k=3, seed=7)
    w2, h2 = nmf_components(x, k=3, seed=7)
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(h1, h2)
    assert (w1 >= 0).all()
    assert (h1 >= 0).all()


def test_nmf_project_recovers_training_rows_approximately() -> None:
    rng = np.random.default_rng(5)
    x = np.abs(rng.normal(loc=3.0, size=(20, 10)))
    w, h = nmf_components(x, k=4, seed=0)
    projected = nmf_project(x, h, seed=0)
    # Projecting the same rows back onto the fixed H should approximately recover W: both
    # reconstruct the same X through the same H, so their residuals should be close.
    np.testing.assert_allclose(projected @ h, w @ h, atol=0.5)


# ==============================================================================================
# random_stand_in
# ==============================================================================================


def test_random_stand_in_shape_determinism_and_seed_difference() -> None:
    a = random_stand_in(50, 20, seed=RANDOM_SEEDS["pca"])
    b = random_stand_in(50, 20, seed=RANDOM_SEEDS["pca"])
    c = random_stand_in(50, 20, seed=RANDOM_SEEDS["nmf"])
    assert a.shape == (50, 20)
    assert a.dtype == np.float64
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


# ==============================================================================================
# linear_kernel
# ==============================================================================================


def test_linear_kernel_mean_diagonal_is_one() -> None:
    rng = np.random.default_rng(6)
    z = rng.normal(size=(10, 5))
    k = linear_kernel(z)
    assert k.shape == (10, 10)
    assert np.diag(k).mean() == pytest.approx(1.0, abs=1e-10)


def test_linear_kernel_raises_on_all_zero_variance_input() -> None:
    z = np.full((10, 3), 5.0)
    with pytest.raises(ValueError, match="all zero"):
        linear_kernel(z)


# ==============================================================================================
# identity_match
# ==============================================================================================


def test_identity_match_perfect_when_halves_are_identical() -> None:
    rng = np.random.default_rng(7)
    half = rng.normal(size=(6, 4))
    assert identity_match(half, half.copy()) == pytest.approx(1.0)


def test_identity_match_is_the_share_of_own_best_match() -> None:
    # Three lines whose profiles are trivially distinct in both halves, so each line's own
    # row is unambiguously its most-correlated match.
    half_a = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    half_b = np.array([[9.0, 0.2, 0.1], [0.2, 9.0, 0.1], [0.1, 0.2, 9.0]])
    assert identity_match(half_a, half_b) == pytest.approx(1.0)

    # Swap two rows of half_b: now lines 0 and 1 each best-match the OTHER line's row, and
    # only line 2 still finds itself -- the share should read exactly 1/3.
    swapped = half_b[[1, 0, 2]]
    assert identity_match(half_a, swapped) == pytest.approx(1.0 / 3.0)


# ==============================================================================================
# chemistry: morgan_fingerprints / tanimoto
# ==============================================================================================


def test_tanimoto_self_one_disjoint_zero_and_matches_rdkit_on_a_hand_pair() -> None:
    smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    fingerprints = morgan_fingerprints(smiles)
    sim = tanimoto(fingerprints)

    np.testing.assert_allclose(np.diag(sim), 1.0)

    # Two fingerprints with no bits in common: force it with synthetic bool rows.
    disjoint = np.zeros((2, 8), dtype=bool)
    disjoint[0, :4] = True
    disjoint[1, 4:] = True
    disjoint_sim = tanimoto(disjoint)
    assert disjoint_sim[0, 1] == 0.0
    assert disjoint_sim[1, 0] == 0.0

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp0 = generator.GetFingerprint(Chem.MolFromSmiles(smiles[0]))
    fp1 = generator.GetFingerprint(Chem.MolFromSmiles(smiles[1]))
    expected = DataStructs.TanimotoSimilarity(fp0, fp1)
    assert sim[0, 1] == pytest.approx(expected, abs=1e-9)


def test_morgan_fingerprints_raises_naming_every_unparseable_smiles() -> None:
    with pytest.raises(ValueError, match="not-a-molecule"):
        morgan_fingerprints(["CCO", "not-a-molecule", "also bad"])


def test_all_107_grid_drugs_smiles_parse() -> None:
    """Every drug the committed, promoted rung 0 table puts in the grid has a SMILES string
    RDKit can parse -- checked through the same crosswalk (`grid_from_rung0` +
    `attach_drug_metadata`) task 2 built, against the committed offline drug-metadata fixture,
    so this is reproducible without any cluster access."""
    per_pair = pd.read_csv(PER_PAIR_PATH, keep_default_na=False, na_values=[""])
    drug_metadata = load_drug_metadata(DRUG_METADATA_PATH)
    grid = grid_from_rung0(per_pair)
    grid = attach_drug_metadata(grid, drug_metadata)
    assert len(grid.drugs) == 107

    smiles_by_name = dict(
        zip(
            drug_metadata["drug"].astype(str),
            drug_metadata["canonical_smiles"].astype(str),
            strict=True,
        )
    )
    smiles = [smiles_by_name[grid.metadata_name[drug]] for drug in grid.drugs]
    fingerprints = morgan_fingerprints(smiles)  # raises if any of the 107 fails to parse
    assert fingerprints.shape == (107, 1024)


# ==============================================================================================
# build control (known answer): identity match recovered / falls to chance under shuffling
# ==============================================================================================


def _synthetic_dmso_cells(
    n_lines: int, n_genes: int, cells_per_line: int, seed: int
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Synthetic DMSO-like cells for ``n_lines`` lines with distinct per-line mean profiles:
    Poisson counts around each line's own mean, halves assigned at random. Fully vectorized
    (no per-cell or per-line Python loop)."""
    rng = np.random.default_rng(seed)
    line_means = rng.uniform(2.0, 400.0, size=(n_lines, n_genes))
    line_idx = np.repeat(np.arange(n_lines), cells_per_line)
    counts_dense = rng.poisson(line_means[line_idx]).astype(np.float32)
    halves = rng.integers(0, 2, size=line_idx.shape[0]).astype(np.int64)
    lines_arr = np.array([f"L{i}" for i in line_idx])
    counts = sparse.csr_matrix(counts_dense)
    return counts, lines_arr, halves


def _fit_describers(
    counts: sparse.csr_matrix, lines_arr: np.ndarray
) -> tuple[list[str], dict[str, object]]:
    """Full-data description parameters (mean/SD/loadings/NMF H) fit from ALL cells (no half
    split), mirroring what the build script fits before describing a half."""
    labels, e_full = pseudobulk_log_cpm(counts, lines_arr)
    mean_e, std_e = standardize_stats(e_full)
    e_std = standardize_apply(e_full, mean_e, std_e)
    _, loadings = pca_components(e_std, k_max=5)
    w, h = nmf_components(e_full, k=4, seed=0)
    mean_w, std_w = standardize_stats(w)
    return list(labels), {
        "mean_e": mean_e,
        "std_e": std_e,
        "loadings": loadings,
        "h": h,
        "mean_w": mean_w,
        "std_w": std_w,
    }


def _half_matrix(
    counts: sparse.csr_matrix,
    lines_arr: np.ndarray,
    halves: np.ndarray,
    half: int,
    categories: list[str],
) -> np.ndarray:
    mask = halves == half
    labels, pseudobulk = pseudobulk_log_cpm(counts[mask], lines_arr[mask])
    frame = pd.DataFrame(pseudobulk, index=pd.Index(labels)).reindex(categories, fill_value=0.0)
    return frame.to_numpy(dtype=np.float64)


@pytest.mark.step_build
def test_build_identity_recovered() -> None:
    """The build step's positive control: a description built from half a line's cells best
    matches that line's other half, for expression, PCA and NMF alike, and each real match
    sits above the shuffled null's 99th percentile (design.md section 8)."""
    n_lines = 50
    counts, lines_arr, halves = _synthetic_dmso_cells(
        n_lines=n_lines, n_genes=40, cells_per_line=300, seed=0
    )
    categories, params = _fit_describers(counts, lines_arr)
    assert len(categories) == n_lines

    mean_e = cast(np.ndarray, params["mean_e"])
    std_e = cast(np.ndarray, params["std_e"])
    loadings = cast(np.ndarray, params["loadings"])
    h = cast(np.ndarray, params["h"])
    mean_w = cast(np.ndarray, params["mean_w"])
    std_w = cast(np.ndarray, params["std_w"])

    def describe_expression(x: np.ndarray) -> np.ndarray:
        return x

    def describe_pca(x: np.ndarray) -> np.ndarray:
        return pca_project(x, mean_e, std_e, loadings)

    def describe_nmf(x: np.ndarray) -> np.ndarray:
        w = nmf_project(x, h, seed=0)
        return standardize_apply(w, mean_w, std_w)

    half0 = _half_matrix(counts, lines_arr, halves, 0, categories)
    half1 = _half_matrix(counts, lines_arr, halves, 1, categories)

    for name, describe in (
        ("expression", describe_expression),
        ("pca", describe_pca),
        ("nmf", describe_nmf),
    ):
        real_share = identity_match(describe(half0), describe(half1))
        assert real_share == pytest.approx(1.0), f"{name}: expected identity 1.0, got {real_share}"

        null = shuffled_identity_null(counts, lines_arr, halves, describe, n_shuffles=50, seed=1)
        p99 = float(np.quantile(null, 0.99))
        assert real_share > p99, f"{name}: real share {real_share} not above null p99 {p99}"


@pytest.mark.step_build
def test_build_shuffled_identity_is_chance() -> None:
    """The build step's negative control: with cells' line labels shuffled first, the mean
    identity share falls to chance (1 / n_lines), within 3 Monte Carlo standard errors of the
    synthetic draw itself (global-constraints invariant 9)."""
    n_lines = 50
    counts, lines_arr, halves = _synthetic_dmso_cells(
        n_lines=n_lines, n_genes=20, cells_per_line=200, seed=2
    )

    def describe_expression(x: np.ndarray) -> np.ndarray:
        return x

    n_shuffles = 50
    null = shuffled_identity_null(
        counts, lines_arr, halves, describe_expression, n_shuffles, seed=3
    )
    se = float(null.std(ddof=1)) / np.sqrt(n_shuffles)
    tol = 3.0 * se
    expected = 1.0 / n_lines
    assert abs(float(null.mean()) - expected) <= tol, (
        f"null mean {null.mean()} not within 3 SE ({tol}) of chance {expected}"
    )


# ==============================================================================================
# CLI end-to-end
# ==============================================================================================


def _write_tiny_cache(cache: Path, lines: list[str], panel: list[str]) -> None:
    import anndata as ad

    rng = np.random.default_rng(0)
    n_lines = len(lines)
    n_genes = len(panel)
    line_means = rng.uniform(2.0, 40.0, size=(n_lines, n_genes))

    cells_dir = cache / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    full_rows: list[np.ndarray] = []
    half_rows: dict[tuple[str, int], np.ndarray] = {}
    for i, line in enumerate(lines):
        n_cells = 40
        counts_dense = rng.poisson(line_means[i], size=(n_cells, n_genes)).astype(np.float32)
        halves = rng.integers(0, 2, size=n_cells).astype(np.int8)
        obs = pd.DataFrame(
            {
                "cellosaurus": [f"CVCL_{i}"] * n_cells,
                "line": [line] * n_cells,
                "plate": ["P1"] * n_cells,
                "key": [f"{k:016x}" for k in range(n_cells)],
                "half": halves,
            },
            # A string obs index (rather than the default RangeIndex) so AnnData does not
            # silently coerce it and warn (anndata.ImplicitModificationWarning).
            index=[f"cell_{i}_{c}" for c in range(n_cells)],
        )
        adata = ad.AnnData(X=sparse.csr_matrix(counts_dense), obs=obs)
        adata.var_names = panel
        adata.var["feature_name"] = panel
        line_path = cells_dir / f"line_{i}.h5ad"
        adata.write_h5ad(line_path)

        totals = counts_dense.sum(axis=1, keepdims=True)
        cpm = np.divide(counts_dense.sum(axis=0), totals.sum()) * 1_000_000.0
        full_rows.append(np.log2(cpm + 1.0))
        for half in (0, 1):
            mask = halves == half
            sub = counts_dense[mask]
            cpm_half = np.divide(sub.sum(axis=0), sub.sum()) * 1_000_000.0
            half_rows[(line, half)] = np.log2(cpm_half + 1.0)

    expr = pd.DataFrame(np.stack(full_rows), index=pd.Index(lines, name="line"), columns=panel)
    expr.to_parquet(cache / "expression.parquet")

    half_index = pd.MultiIndex.from_tuples(
        [(line, half) for line in lines for half in (0, 1)], names=["line", "half"]
    )
    half_values = np.stack([half_rows[key] for key in half_index])
    expr_halves = pd.DataFrame(half_values, index=half_index, columns=panel)
    expr_halves.to_parquet(cache / "expression_halves.parquet")


@pytest.mark.step_build
def test_cli_writes_every_output_and_its_record(tmp_path: Path) -> None:
    # NMF's "nndsvda" init requires n_components <= min(n_samples, n_features); the largest k
    # this task uses is 20, so both lines and the gene panel must be at least that wide.
    lines = [f"L{i}" for i in range(22)]
    panel = [f"G{j}" for j in range(25)]
    cache = tmp_path / "cache"
    _write_tiny_cache(cache, lines, panel)

    drugs = ["DrugA", "DrugB", "DrugC"]
    smiles = {"DrugA": "CCO", "DrugB": "CC(=O)O", "DrugC": "c1ccccc1"}
    drug_metadata_path = tmp_path / "drug_metadata.csv"
    pd.DataFrame(
        {
            "drug": drugs,
            "pubchem_cid": ["1", "2", "3"],
            "canonical_smiles": [smiles[d] for d in drugs],
        }
    ).to_csv(drug_metadata_path, index=False)

    grid_path = tmp_path / "rung1_grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "lines": lines,
                "drugs": drugs,
                "metadata_name": {d: d for d in drugs},
                "excluded_pairs": [],
            }
        )
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(REPO / "scripts/heldout_descriptions.py"),
        "--grid",
        str(grid_path),
        "--drug-metadata",
        str(drug_metadata_path),
        "--cache",
        str(cache),
        "--out-dir",
        str(out_dir),
        "--n-shuffles",
        "5",
        "--seed",
        "0",
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert first.returncode == 0

    descriptions_path = cache / "descriptions.npz"
    tanimoto_path = cache / "tanimoto.npz"
    identity_match_path = out_dir / "rung1_identity_match.csv"

    assert is_done(descriptions_path)
    assert is_done(tanimoto_path)
    assert is_done(identity_match_path)

    n_lines = len(lines)
    n_genes = len(panel)
    npz = np.load(descriptions_path)
    assert list(npz["lines"]) == lines
    assert npz["expression"].shape == (n_lines, n_genes)
    assert npz["pca"].shape[0] == n_lines
    assert npz["pca_loadings"].shape[1] == n_genes
    for k in (2, 5, 10, 15, 20):
        assert npz[f"nmf_{k}"].shape == (n_lines, k)
    assert npz["random_expression"].shape == (n_lines, n_genes)
    assert npz["random_pca"].shape == (n_lines, 20)
    assert npz["random_nmf"].shape == (n_lines, 20)

    # Every stored description is standardized across lines: column means ~ 0, column SDs ~ 1
    # (a zero-variance column would be exactly 0, but none arise for this fixture).
    for name in ("expression", "pca", *(f"nmf_{k}" for k in (2, 5, 10, 15, 20))):
        arr = npz[name]
        np.testing.assert_allclose(arr.mean(axis=0), 0.0, atol=1e-8, err_msg=f"{name} mean")
        np.testing.assert_allclose(arr.std(axis=0, ddof=0), 1.0, atol=1e-6, err_msg=f"{name} sd")

    tan = np.load(tanimoto_path)
    assert list(tan["drugs"]) == drugs
    assert tan["similarity"].shape == (3, 3)
    assert tan["fingerprints"].shape == (3, 1024)
    np.testing.assert_allclose(np.diag(tan["similarity"]), 1.0)

    identity = pd.read_csv(identity_match_path)
    assert list(identity.columns) == [
        "description",
        "identity_share",
        "null_mean",
        "null_p99",
        "n_lines",
        "n_shuffles",
    ]
    assert sorted(identity["description"]) == ["expression", "nmf", "pca"]
    assert (identity["n_lines"] == n_lines).all()
    assert (identity["n_shuffles"] == 5).all()

    for name in ("expression", "pca", "nmf"):
        grid_csv_path = out_dir / f"rung1_identity_grid_{name}.csv"
        assert is_done(grid_csv_path)
        long = pd.read_csv(grid_csv_path)
        assert list(long.columns) == ["line_a", "line_b", "r", "source"]
        assert set(long["source"]) == {"real", "shuffled"}

    # Reruns skip already-done outputs.
    mtime_before = descriptions_path.stat().st_mtime_ns
    second = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "skipping" in second.stdout
    assert descriptions_path.stat().st_mtime_ns == mtime_before
    assert second.returncode == 0
