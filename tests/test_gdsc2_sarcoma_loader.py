"""Tests for fmharness.data.loaders.gdsc2_sarcoma.

Hermetic: builds a synthetic data/raw/gdsc2_sarcoma/ tree with miniature
Model.csv, GDSC2 dose-response xlsx, DepMap raw-counts CSV, and a
manifest.json with real shas. Also writes a tiny drug_xref.parquet in
data/static/. Does not depend on the real ~129 MB DepMap file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fmharness.data.loaders.gdsc2_sarcoma import (
    IngestError,
    load_gdsc2_sarcoma,
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_fixture(repo: Path) -> None:
    """Populate repo/data/raw/gdsc2_sarcoma/ and repo/data/static/ with mini fixtures."""
    raw = repo / "data" / "raw" / "gdsc2_sarcoma"
    (raw / "depmap").mkdir(parents=True)
    (raw / "gdsc2").mkdir(parents=True)
    static = repo / "data" / "static"
    static.mkdir(parents=True)

    # --- Model.csv: 5 lines (2 sarcoma w/ COSMIC, 1 sarcoma w/o COSMIC, 2 non-sarcoma) ---
    model_df = pd.DataFrame(
        [
            {
                "ModelID": "ACH-001",
                "COSMICID": 100001,
                "OncotreeLineage": "Bone",
                "OncotreeSubtype": "Ewing Sarcoma",
                "CCLEName": "EW01",
                "SangerModelID": "SIDM001",
            },
            {
                "ModelID": "ACH-002",
                "COSMICID": 100002,
                "OncotreeLineage": "Soft Tissue",
                "OncotreeSubtype": "Leiomyosarcoma",
                "CCLEName": "LMS01",
                "SangerModelID": "SIDM002",
            },
            {
                "ModelID": "ACH-003",
                "COSMICID": None,
                "OncotreeLineage": "Bone",
                "OncotreeSubtype": "Osteosarcoma",
                "CCLEName": "OS01",
                "SangerModelID": "SIDM003",
            },
            {
                "ModelID": "ACH-004",
                "COSMICID": 100004,
                "OncotreeLineage": "Lung",
                "OncotreeSubtype": "NSCLC",
                "CCLEName": "LU01",
                "SangerModelID": "SIDM004",
            },
            {
                "ModelID": "ACH-005",
                "COSMICID": 100005,
                "OncotreeLineage": "Bone",
                "OncotreeSubtype": "Ewing Sarcoma",
                "CCLEName": "EW02",
                "SangerModelID": "SIDM005",
            },
        ]
    )
    model_df.to_csv(raw / "depmap" / "Model.csv", index=False)

    # --- DepMap raw counts: 4 lines (ACH-001, ACH-002, ACH-004 = sarcoma+lung; ACH-005 = sarcoma
    # but as IsDefaultEntryForModel="No" to exercise the dedupe filter). ACH-003 is intentionally
    # absent (no RNA-seq) to verify the cohort shrinks to RNA-sampled lines. 3 genes. ---
    expr_df = pd.DataFrame(
        [
            {
                "SequencingID": "CDS-A",
                "ModelConditionID": "MC-A",
                "ModelID": "ACH-001",
                "IsDefaultEntryForMC": "Yes",
                "IsDefaultEntryForModel": "Yes",
                "TSPAN6 (7105)": 500,
                "TNMD (64102)": 10,
                "DPM1 (8813)": 1200,
            },
            {
                "SequencingID": "CDS-B",
                "ModelConditionID": "MC-B",
                "ModelID": "ACH-002",
                "IsDefaultEntryForMC": "Yes",
                "IsDefaultEntryForModel": "Yes",
                "TSPAN6 (7105)": 700,
                "TNMD (64102)": 25,
                "DPM1 (8813)": 1500,
            },
            {
                "SequencingID": "CDS-C",
                "ModelConditionID": "MC-C",
                "ModelID": "ACH-004",
                "IsDefaultEntryForMC": "Yes",
                "IsDefaultEntryForModel": "Yes",
                "TSPAN6 (7105)": 400,
                "TNMD (64102)": 5,
                "DPM1 (8813)": 900,
            },
            {
                "SequencingID": "CDS-D",
                "ModelConditionID": "MC-D",
                "ModelID": "ACH-005",
                "IsDefaultEntryForMC": "No",
                "IsDefaultEntryForModel": "No",
                "TSPAN6 (7105)": 999,
                "TNMD (64102)": 999,
                "DPM1 (8813)": 999,
            },
        ]
    )
    # Save with a leading unnamed index column to match the real DepMap CSV shape
    expr_df.to_csv(raw / "depmap" / "OmicsExpressionRawReadCountHumanProteinCodingGenes.csv")

    # --- GDSC2 dose-response: 3 lines x 2 drugs = 6 rows. ACH-001 + ACH-002 are sarcoma,
    # ACH-004 is lung (excluded by sarcoma filter). ACH-005 has GDSC2 entry but no RNA-seq
    # default-row, so excluded by RNA-seq availability ---
    gdsc_df = pd.DataFrame(
        [
            # Imatinib for ACH-001
            {
                "COSMIC_ID": 100001,
                "DRUG_ID": "1005",
                "DRUG_NAME": "Imatinib",
                "LN_IC50": -1.5,
                "AUC": 0.72,
            },
            # Imatinib for ACH-002
            {
                "COSMIC_ID": 100002,
                "DRUG_ID": "1005",
                "DRUG_NAME": "Imatinib",
                "LN_IC50": 0.3,
                "AUC": 0.55,
            },
            # SomeNewCompound (unresolvable in xref) for ACH-001
            {
                "COSMIC_ID": 100001,
                "DRUG_ID": "9999",
                "DRUG_NAME": "SomeNewCompound",
                "LN_IC50": 2.1,
                "AUC": 0.21,
            },
            # Imatinib for ACH-004 (lung, excluded by sarcoma filter)
            {
                "COSMIC_ID": 100004,
                "DRUG_ID": "1005",
                "DRUG_NAME": "Imatinib",
                "LN_IC50": -0.8,
                "AUC": 0.66,
            },
            # Imatinib for ACH-005 (sarcoma but no RNA-seq default-row → excluded)
            {
                "COSMIC_ID": 100005,
                "DRUG_ID": "1005",
                "DRUG_NAME": "Imatinib",
                "LN_IC50": 0.0,
                "AUC": 0.5,
            },
        ]
    )
    gdsc_df.to_excel(
        raw / "gdsc2" / "GDSC2_fitted_dose_response_27Oct23.xlsx", sheet_name="Sheet1", index=False
    )

    # --- Manifest: real shas of all files ---
    paths = {
        "depmap/Model.csv": raw / "depmap" / "Model.csv",
        "depmap/OmicsExpressionRawReadCountHumanProteinCodingGenes.csv": raw
        / "depmap"
        / "OmicsExpressionRawReadCountHumanProteinCodingGenes.csv",
        "gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx": raw
        / "gdsc2"
        / "GDSC2_fitted_dose_response_27Oct23.xlsx",
    }
    manifest = {
        "dataset": "gdsc2_sarcoma",
        "release": {"gdsc": "8.5", "depmap": "26Q1"},
        "files": {
            rel: {"sha256": _sha(p), "bytes": p.stat().st_size, "source_uri": "test://fixture"}
            for rel, p in paths.items()
        },
    }
    (raw / "manifest.json").write_text(json.dumps(manifest))

    # --- drug_xref.parquet: Imatinib resolves, SomeNewCompound does not ---
    xref = pd.DataFrame(
        [
            {
                "input_name": "Imatinib",
                "source": "gdsc2",
                "source_drug_id": "1005",
                "pubchem_cid": 5291,
                "inchikey": "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
                "drugbank_id": "DB00619",
                "resolution_method": "gdsc2_pubchem_column",
                "notes": None,
            },
            {
                "input_name": "SomeNewCompound",
                "source": "gdsc2",
                "source_drug_id": "9999",
                "pubchem_cid": None,
                "inchikey": None,
                "drugbank_id": None,
                "resolution_method": "unresolved",
                "notes": None,
            },
        ]
    )
    xref["pubchem_cid"] = xref["pubchem_cid"].astype("Int64")
    xref_path = static / "drug_xref.parquet"
    xref.to_parquet(xref_path, index=False)
    static_manifest = {
        "dataset": "fmharness_static_assets",
        "release": {"asset_set": "drug_xref_test"},
        "files": {
            "drug_xref.parquet": {
                "sha256": _sha(xref_path),
                "bytes": xref_path.stat().st_size,
                "source_uri": "test://fixture",
            }
        },
    }
    (static / "manifest.json").write_text(json.dumps(static_manifest))


def test_loader_end_to_end(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_gdsc2_sarcoma(tmp_path, ingestion_date=date(2026, 5, 25))

    # Cohort: sarcoma lines with COSMIC AND in GDSC2 AND in DepMap RNA-seq default rows.
    # Expected: ACH-001 + ACH-002 (ACH-003 no COSMIC, ACH-004 not sarcoma, ACH-005 no default RNA-seq row).
    assert bundle.tranche.patient_count == 2
    assert bundle.tranche.sample_count == 2
    assert bundle.tranche.drug_count == 2  # Imatinib + SomeNewCompound
    assert set(bundle.tranche.subtypes) == {"Ewing Sarcoma", "Leiomyosarcoma"}
    assert len(bundle.tranche.content_hash) == 64  # sha256 hex

    # Expression: 2 cells x 3 genes
    assert bundle.expression.shape == (2, 3)
    assert list(bundle.expression.var.index) == ["TSPAN6", "TNMD", "DPM1"]
    assert "raw_counts" in bundle.expression.layers
    # DESeq2 normalized values are floats; raw counts are ints
    X = bundle.expression.X
    assert X is not None
    assert X.dtype.kind == "f"
    assert bundle.expression.layers["raw_counts"].dtype.kind == "i"
    # size_factor populated per sample
    assert "size_factor" in bundle.expression.obs.columns
    assert (bundle.expression.obs["size_factor"] > 0).all()


def test_drug_assays_two_metrics_per_pair(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_gdsc2_sarcoma(tmp_path)

    # 2 cells * 2 drugs (Imatinib for both, SomeNewCompound only for ACH-001)
    # = 3 (cell, drug) pairs * 2 metrics = 6 assay rows
    assert len(bundle.drug_assays) == 6
    ic50 = [a for a in bundle.drug_assays if a.response_metric == "ic50"]
    auc = [a for a in bundle.drug_assays if a.response_metric == "auc"]
    assert len(ic50) == 3
    assert len(auc) == 3

    # Lung line (ACH-004) excluded
    assert not any(a.sample_id == "ACH-004" for a in bundle.drug_assays)
    # No-RNA-seq sarcoma line (ACH-005) excluded
    assert not any(a.sample_id == "ACH-005" for a in bundle.drug_assays)


def test_xref_attachment(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_gdsc2_sarcoma(tmp_path)

    imatinib_rows = [a for a in bundle.drug_assays if a.drug_name == "Imatinib"]
    assert all(a.pubchem_cid == 5291 for a in imatinib_rows)
    assert all(a.inchikey == "KTUFNOKKBVMGRW-UHFFFAOYSA-N" for a in imatinib_rows)
    assert all(a.drugbank_id == "DB00619" for a in imatinib_rows)

    unresolved = [a for a in bundle.drug_assays if a.drug_name == "SomeNewCompound"]
    assert unresolved  # at least one
    assert all(a.pubchem_cid is None for a in unresolved)
    assert all(a.inchikey is None for a in unresolved)


def test_sarcoma_subtype_metadata(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_gdsc2_sarcoma(tmp_path)

    by_id = {p.patient_id: p for p in bundle.patients}
    assert by_id["ACH-001"].subtype == "Ewing Sarcoma"
    assert by_id["ACH-001"].tissue_of_origin == "Bone"
    assert by_id["ACH-001"].subtype_granularity == "fine"
    assert by_id["ACH-001"].metadata["cosmic_id"] == 100001
    assert by_id["ACH-001"].metadata["ccle_name"] == "EW01"
    assert by_id["ACH-002"].subtype == "Leiomyosarcoma"
    assert by_id["ACH-002"].tissue_of_origin == "Soft Tissue"


def test_baseline_expression_per_sample(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_gdsc2_sarcoma(tmp_path)

    assert len(bundle.baseline_expression) == 2
    for be in bundle.baseline_expression:
        assert be.normalization == "median_of_ratios"
        assert be.gene_id_namespace == "symbol"
        assert be.gene_count == 3
        assert be.reference_genome == "GRCh38"
        assert "expression.h5ad" in be.expression_matrix_uri


def test_content_hash_deterministic(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    h1 = load_gdsc2_sarcoma(tmp_path).tranche.content_hash
    h2 = load_gdsc2_sarcoma(tmp_path).tranche.content_hash
    assert h1 == h2


def test_manifest_mismatch_refused(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    # Corrupt the recorded sha for one file
    raw = tmp_path / "data" / "raw" / "gdsc2_sarcoma"
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest["files"]["depmap/Model.csv"]["sha256"] = "0" * 64
    (raw / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(IngestError, match="sha256 mismatch"):
        load_gdsc2_sarcoma(tmp_path)


def test_manifest_missing_refused(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    (tmp_path / "data" / "raw" / "gdsc2_sarcoma" / "manifest.json").unlink()
    with pytest.raises(IngestError, match="raw manifest missing"):
        load_gdsc2_sarcoma(tmp_path)


def test_skip_verify_allows_corrupt_manifest(tmp_path: Path) -> None:
    """verify_manifest=False is an escape hatch for tests / fast paths."""
    _build_fixture(tmp_path)
    raw = tmp_path / "data" / "raw" / "gdsc2_sarcoma"
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest["files"]["depmap/Model.csv"]["sha256"] = "0" * 64
    (raw / "manifest.json").write_text(json.dumps(manifest))

    # Should NOT raise -- but the content_hash will still be computed from the
    # (now incorrect) manifest values, which is the documented behavior.
    bundle = load_gdsc2_sarcoma(tmp_path, verify_manifest=False)
    assert bundle.tranche.patient_count == 2
