"""Tests for fmharness.data.loaders.soragni.

Hermetic: builds a synthetic data/raw/soragni/tables/ tree (normalized counts,
drug screen, manifest) plus a small data/static/drug_xref.parquet. Covers the
sample-ID parsing edge cases (letter suffixes, timepoints), the matched-cohort
intersection, drug-assay attachment to Organoid samples, and manifest refusal.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fmharness.data.loaders.soragni import (
    IngestError,
    canonicalize_patient_id,
    load_soragni,
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_fixture(repo: Path) -> None:
    raw = repo / "data" / "raw" / "soragni" / "tables"
    raw.mkdir(parents=True)
    static = repo / "data" / "static"
    static.mkdir(parents=True)

    # --- normalized_gene_counts: 5 genes x sample columns.
    # Patients:
    #   SARC0001  -- has Tumor + Organoid + in drug_screen
    #   SARC0002  -- has Tumor + Organoid + in drug_screen (timepoint A)
    #   SARC0003  -- has Tumor + Organoid but NOT in drug_screen (RNA-only)
    #   SARC0004_2 -- has Tumor + Organoid + in drug_screen (timepoint suffix)
    # Plus "col" and "col1" straggler columns (must be ignored).
    ngc_rows = [
        {
            "Chromosome": "1",
            "Start": 100,
            "Stop": 200,
            "Length": 100,
            "Strand": "+",
            "Gene_Symbol": "GENE1",
            "gene_biotype": "protein_coding",
            "gene_id": "ENSG00000000001",
            "gene_name": "GENE1",
            "gene_source": "ensembl_havana",
            "gene_version": "1",
            "SARC0001_Tumor": 100,
            "SARC0001_Organoids": 110,
            "SARC0002_Tumor": 200,
            "SARC0002_Organoids": 220,
            "SARC0003_Tumor": 50,
            "SARC0003_Organoids": 55,
            "SARC0004_2_Tumor": 80,
            "SARC0004_2_Organoids": 90,
            "col": None,
            "col1": None,
        },
        {
            "Chromosome": "2",
            "Start": 300,
            "Stop": 400,
            "Length": 100,
            "Strand": "+",
            "Gene_Symbol": "GENE2",
            "gene_biotype": "lncRNA",
            "gene_id": "ENSG00000000002",
            "gene_name": "GENE2",
            "gene_source": "ensembl_havana",
            "gene_version": "1",
            "SARC0001_Tumor": 10,
            "SARC0001_Organoids": 12,
            "SARC0002_Tumor": 20,
            "SARC0002_Organoids": 25,
            "SARC0003_Tumor": 5,
            "SARC0003_Organoids": 6,
            "SARC0004_2_Tumor": 8,
            "SARC0004_2_Organoids": 9,
            "col": None,
            "col1": None,
        },
    ]
    ngc = pd.DataFrame(ngc_rows)
    ngc.to_parquet(raw / "normalized_gene_counts.parquet", index=False)

    # --- drug_screen: includes the matched patients + an unmatched patient with letter suffix
    # (which must NOT crash the canonicalizer)
    drug_rows = [
        {
            "Sample_ID": "sarc0001",
            "Diagnosis": "osteosarcoma",
            "Tumor_Type": "primary",
            "Age_Category": "Adult",
            "Drug_Name": "Imatinib",
            "Viability_Score": 45.2,
        },
        {
            "Sample_ID": "sarc0001",
            "Diagnosis": "osteosarcoma",
            "Tumor_Type": "primary",
            "Age_Category": "Adult",
            "Drug_Name": "Topotecan",
            "Viability_Score": 12.0,
        },
        {
            "Sample_ID": "sarc0002",
            "Diagnosis": "leiomyosarcoma",
            "Tumor_Type": "primary",
            "Age_Category": "Adult",
            "Drug_Name": "Imatinib",
            "Viability_Score": 88.4,
        },
        {
            "Sample_ID": "sarc0004_2",
            "Diagnosis": "synovial sarcoma",
            "Tumor_Type": "recurrence",
            "Age_Category": "Adult",
            "Drug_Name": "Imatinib",
            "Viability_Score": 100.0,
        },
        {
            "Sample_ID": "sarc0004_2",
            "Diagnosis": "synovial sarcoma",
            "Tumor_Type": "recurrence",
            "Age_Category": "Adult",
            "Drug_Name": "UnresolvedCompound",
            "Viability_Score": 70.0,
        },
        # Letter-suffix patient (mirrors real "sarc0028_biopsy", "sarc0024_B" etc.)
        # Not in RNA-seq cohort -> excluded but must NOT crash the canonicalizer.
        {
            "Sample_ID": "sarc0099_biopsy",
            "Diagnosis": "ewing sarcoma",
            "Tumor_Type": "primary",
            "Age_Category": "Pediatric",
            "Drug_Name": "Imatinib",
            "Viability_Score": 50.0,
        },
    ]
    pd.DataFrame(drug_rows).to_parquet(raw / "drug_screen.parquet", index=False)

    # --- Manifest: real shas
    paths = {
        "normalized_gene_counts.parquet": raw / "normalized_gene_counts.parquet",
        "drug_screen.parquet": raw / "drug_screen.parquet",
    }
    manifest = {
        "dataset": "soragni_pdo_sarcoma_2024",
        "release": {"project": "syn55180195", "mode": "tables"},
        "files": {
            rel: {"sha256": _sha(p), "bytes": p.stat().st_size, "source_uri": "test://fixture"}
            for rel, p in paths.items()
        },
    }
    (raw / "manifest.json").write_text(json.dumps(manifest))

    # --- drug_xref: Imatinib and Topotecan resolve, UnresolvedCompound does not
    xref = pd.DataFrame(
        [
            {
                "input_name": "Imatinib",
                "source": "soragni",
                "source_drug_id": None,
                "pubchem_cid": 5291,
                "inchikey": "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
                "drugbank_id": "DB00619",
                "resolution_method": "soragni_via_gdsc2_synonym",
                "notes": None,
            },
            {
                "input_name": "Topotecan",
                "source": "soragni",
                "source_drug_id": None,
                "pubchem_cid": 60700,
                "inchikey": "UCFGDBYHRUNTLO-QHCPKHFHSA-N",
                "drugbank_id": "DB01030",
                "resolution_method": "soragni_via_gdsc2_synonym",
                "notes": None,
            },
            {
                "input_name": "UnresolvedCompound",
                "source": "soragni",
                "source_drug_id": None,
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


def test_canonicalize_patient_id_variants() -> None:
    # Basic numeric IDs
    assert canonicalize_patient_id("sarc0001") == "SARC0001"
    assert canonicalize_patient_id("SARC0001") == "SARC0001"
    # Timepoint suffix (numeric)
    assert canonicalize_patient_id("SARC0139_1") == "SARC0139_1"
    assert canonicalize_patient_id("sarc0004_2") == "SARC0004_2"
    # Specimen suffix is stripped
    assert canonicalize_patient_id("SARC0065_Tumor") == "SARC0065"
    assert canonicalize_patient_id("SARC0095_Organoids") == "SARC0095"
    assert canonicalize_patient_id("SARC0139_1_Tumor") == "SARC0139_1"
    # Letter / word suffix (real data has sarc0024_B, sarc0028_biopsy, sarc0053_a/b/c)
    assert canonicalize_patient_id("sarc0024_B") == "SARC0024_B"
    assert canonicalize_patient_id("sarc0028_biopsy") == "SARC0028_BIOPSY"
    assert canonicalize_patient_id("sarc0053_a") == "SARC0053_A"


def test_canonicalize_patient_id_rejects_non_soragni() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        canonicalize_patient_id("ACH-001113")
    with pytest.raises(ValueError, match="unrecognized"):
        canonicalize_patient_id("not_a_sample_id")


def test_loader_end_to_end(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_soragni(tmp_path, ingestion_date=date(2026, 5, 25))

    # Cohort: SARC0001 + SARC0002 + SARC0004_2 (SARC0003 has RNA but no drug screen)
    assert bundle.tranche.patient_count == 3
    assert bundle.tranche.sample_count == 6  # 3 patients x (Tumor + Organoid)
    assert bundle.tranche.drug_count == 3  # Imatinib + Topotecan + UnresolvedCompound
    assert set(bundle.tranche.subtypes) == {"osteosarcoma", "leiomyosarcoma", "synovial sarcoma"}
    assert len(bundle.tranche.content_hash) == 64

    # Expression: 6 cells x 2 genes
    assert bundle.expression.shape == (6, 2)
    # var indexed by Ensembl gene_id
    assert list(bundle.expression.var.index) == ["ENSG00000000001", "ENSG00000000002"]
    assert set(bundle.expression.var.columns) >= {"Gene_Symbol", "gene_biotype"}
    # Sample IDs are <patient>_<specimen capitalized>
    assert sorted(bundle.expression.obs.index) == [
        "SARC0001_Organoid",
        "SARC0001_Tumor",
        "SARC0002_Organoid",
        "SARC0002_Tumor",
        "SARC0004_2_Organoid",
        "SARC0004_2_Tumor",
    ]


def test_stragglers_ignored(tmp_path: Path) -> None:
    """The 'col' and 'col1' columns in the real Soragni table must NOT appear as samples."""
    _build_fixture(tmp_path)
    bundle = load_soragni(tmp_path)
    assert not any(s.sample_id in ("col", "col1") for s in bundle.samples)
    assert "col" not in bundle.expression.obs.index
    assert "col1" not in bundle.expression.obs.index


def test_drug_assays_attached_to_organoid(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_soragni(tmp_path)

    # 6 drug-screen rows in fixture; 5 in the matched cohort (sarc0099 excluded)
    assert len(bundle.drug_assays) == 5
    # All assays attach to an Organoid sample, never a Tumor sample
    for a in bundle.drug_assays:
        assert a.sample_id.endswith("_Organoid")
        assert a.response_metric == "viability"

    # No assay for the excluded patient (sarc0099_biopsy, not in RNA-seq cohort)
    assert not any(a.sample_id.startswith("SARC0099") for a in bundle.drug_assays)


def test_xref_attachment(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_soragni(tmp_path)

    imatinib = [a for a in bundle.drug_assays if a.drug_name == "Imatinib"]
    assert imatinib  # at least one
    assert all(a.pubchem_cid == 5291 for a in imatinib)
    assert all(a.inchikey == "KTUFNOKKBVMGRW-UHFFFAOYSA-N" for a in imatinib)

    unresolved = [a for a in bundle.drug_assays if a.drug_name == "UnresolvedCompound"]
    assert unresolved
    assert all(a.pubchem_cid is None for a in unresolved)


def test_baseline_expression_per_sample(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_soragni(tmp_path)

    # One BaselineExpression per sample (6 total)
    assert len(bundle.baseline_expression) == 6
    for be in bundle.baseline_expression:
        assert be.normalization == "cpm"
        assert be.gene_id_namespace == "ensembl"
        assert be.gene_count == 2
        assert "expression.h5ad" in be.expression_matrix_uri


def test_patient_subtype_from_drug_screen(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    bundle = load_soragni(tmp_path)

    by_id = {p.patient_id: p for p in bundle.patients}
    assert by_id["SARC0001"].subtype == "osteosarcoma"
    assert by_id["SARC0001"].subtype_granularity == "fine"
    assert by_id["SARC0001"].tissue_of_origin == "sarcoma"
    assert by_id["SARC0004_2"].subtype == "synovial sarcoma"


def test_content_hash_deterministic(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    h1 = load_soragni(tmp_path).tranche.content_hash
    h2 = load_soragni(tmp_path).tranche.content_hash
    assert h1 == h2


def test_manifest_mismatch_refused(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    raw = tmp_path / "data" / "raw" / "soragni" / "tables"
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest["files"]["drug_screen.parquet"]["sha256"] = "0" * 64
    (raw / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(IngestError, match="sha256 mismatch"):
        load_soragni(tmp_path)


def test_manifest_missing_refused(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    (tmp_path / "data" / "raw" / "soragni" / "tables" / "manifest.json").unlink()
    with pytest.raises(IngestError, match="raw manifest missing"):
        load_soragni(tmp_path)


def test_skip_verify_allows_corrupt_manifest(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    raw = tmp_path / "data" / "raw" / "soragni" / "tables"
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest["files"]["drug_screen.parquet"]["sha256"] = "0" * 64
    (raw / "manifest.json").write_text(json.dumps(manifest))

    # Loads despite the corrupt manifest sha (escape hatch for tests / fast paths).
    bundle = load_soragni(tmp_path, verify_manifest=False)
    assert bundle.tranche.patient_count == 3
