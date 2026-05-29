"""Tests for fmharness.data.loaders.coderdata.

Hermetic: monkey-patches ``coderdata.load`` to return a synthetic
``Dataset`` built from in-memory DataFrames. No figshare downloads; no
dependency on the real CoderData files.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import coderdata as cd
import pandas as pd
import pytest

from fmharness.data.loaders.coderdata import IngestError, load_coderdata_tranche


def _build_dataset() -> cd.Dataset:
    """Synthetic CoderData Dataset mirroring sarcoma's table shape.

    - 4 patients (P01, P02, P03, P04). P01-P03 are matched (RNA + experiments);
      P04 has experiments but no RNA (must be excluded from the cohort).
    - Each matched patient has Tumor + Organoid samples; experiments attach
      to the Tumor sample only.
    - 3 genes; 3 drugs.
    """
    samples = pd.DataFrame(
        [
            {
                "other_id": "P01_Tumor",
                "common_name": "P01",
                "other_id_source": "Synapse",
                "other_names": None,
                "cancer_type": "Osteosarcoma",
                "species": "Homo sapiens(Human)",
                "model_type": "tumor",
                "improve_sample_id": 1001,
            },
            {
                "other_id": "P01_Organoid",
                "common_name": "P01",
                "other_id_source": "Synapse",
                "other_names": None,
                "cancer_type": "Osteosarcoma",
                "species": "Homo sapiens(Human)",
                "model_type": "patient derived organoid",
                "improve_sample_id": 1002,
            },
            {
                "other_id": "P02_Tumor",
                "common_name": "P02",
                "other_id_source": "Synapse",
                "other_names": None,
                "cancer_type": "Ewing Sarcoma",
                "species": "Homo sapiens(Human)",
                "model_type": "tumor",
                "improve_sample_id": 1003,
            },
            {
                "other_id": "P02_Organoid",
                "common_name": "P02",
                "other_id_source": "Synapse",
                "other_names": None,
                "cancer_type": "Ewing Sarcoma",
                "species": "Homo sapiens(Human)",
                "model_type": "patient derived organoid",
                "improve_sample_id": 1004,
            },
            {
                "other_id": "P03_Tumor",
                "common_name": "P03",
                "other_id_source": "Synapse",
                "other_names": None,
                "cancer_type": "Leiomyosarcoma",
                "species": "Homo sapiens(Human)",
                "model_type": "tumor",
                "improve_sample_id": 1005,
            },
            {
                "other_id": "P03_Organoid",
                "common_name": "P03",
                "other_id_source": "Synapse",
                "other_names": None,
                "cancer_type": "Leiomyosarcoma",
                "species": "Homo sapiens(Human)",
                "model_type": "patient derived organoid",
                "improve_sample_id": 1006,
            },
            # P04: experiments only, no RNA (must be excluded from matched cohort)
            {
                "other_id": "P04_Tumor",
                "common_name": "P04",
                "other_id_source": "Synapse",
                "other_names": None,
                "cancer_type": "Chondrosarcoma",
                "species": "Homo sapiens(Human)",
                "model_type": "tumor",
                "improve_sample_id": 1007,
            },
        ]
    )

    # Transcriptomics: P01-P03 each have RNA on Tumor + Organoid; P04 has none.
    # 3 genes per sample.
    tx_rows = []
    for sid in (1001, 1002, 1003, 1004, 1005, 1006):
        for entrez, value in [(7105, 10.0 + sid * 0.01), (1000, 5.0), (9999, 0.5)]:
            tx_rows.append(
                {
                    "entrez_id": entrez,
                    "improve_sample_id": sid,
                    "transcriptomics": value,
                    "source": "Synapse",
                    "study": "Test",
                }
            )
    transcriptomics = pd.DataFrame(tx_rows)

    drugs = pd.DataFrame(
        [
            {
                "improve_drug_id": "D01",
                "chem_name": "imatinib",
                "pubchem_id": 5291,
                "canSMILES": "C",
                "InChIKey": "INCHI-IMA-X",
                "formula": "C29",
                "weight": 493.6,
            },
            {
                "improve_drug_id": "D02",
                "chem_name": "topotecan",
                "pubchem_id": 60700,
                "canSMILES": "C",
                "InChIKey": "INCHI-TOP-X",
                "formula": "C23",
                "weight": 421.4,
            },
            {
                "improve_drug_id": "D03",
                "chem_name": "unresolved_drug",
                "pubchem_id": pd.NA,
                "canSMILES": "C",
                "InChIKey": pd.NA,
                "formula": pd.NA,
                "weight": pd.NA,
            },
        ]
    )

    # Experiments: attach to Tumor samples (1001, 1003, 1005, 1007).
    # P01: 2 drugs x auc; P02: 1 drug x auc; P03: 1 drug x ic50; P04: 1 drug x auc.
    experiments = pd.DataFrame(
        [
            {
                "source": "Test",
                "improve_sample_id": 1001,
                "improve_drug_id": "D01",
                "study": "Test",
                "time": 2,
                "time_unit": "days",
                "dose_response_metric": "published_auc",
                "dose_response_value": 0.85,
            },
            {
                "source": "Test",
                "improve_sample_id": 1001,
                "improve_drug_id": "D02",
                "study": "Test",
                "time": 2,
                "time_unit": "days",
                "dose_response_metric": "published_auc",
                "dose_response_value": 0.45,
            },
            {
                "source": "Test",
                "improve_sample_id": 1003,
                "improve_drug_id": "D03",
                "study": "Test",
                "time": 2,
                "time_unit": "days",
                "dose_response_metric": "published_auc",
                "dose_response_value": 1.10,
            },
            {
                "source": "Test",
                "improve_sample_id": 1005,
                "improve_drug_id": "D01",
                "study": "Test",
                "time": 2,
                "time_unit": "days",
                "dose_response_metric": "fit_ic50",
                "dose_response_value": 2.50,
            },
            # Skipped metric: should be dropped silently
            {
                "source": "Test",
                "improve_sample_id": 1005,
                "improve_drug_id": "D02",
                "study": "Test",
                "time": 2,
                "time_unit": "days",
                "dose_response_metric": "fit_hs",
                "dose_response_value": 1.2,
            },
            # P04 -- has experiments but no RNA (excluded by matched-cohort filter)
            {
                "source": "Test",
                "improve_sample_id": 1007,
                "improve_drug_id": "D01",
                "study": "Test",
                "time": 2,
                "time_unit": "days",
                "dose_response_metric": "published_auc",
                "dose_response_value": 0.99,
            },
        ]
    )

    return cd.Dataset(
        name="test_synthetic",
        samples=samples,
        transcriptomics=transcriptomics,
        drugs=drugs,
        experiments=experiments,
    )


@pytest.fixture
def _patched_cd(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Monkey-patch ``cd.download`` (no-op) and ``cd.load`` (return synthetic)."""
    ds = _build_dataset()
    monkeypatch.setattr(cd, "download", lambda **_: None)
    monkeypatch.setattr(cd, "load", lambda *_a, **_kw: ds)


def test_basic_matched_cohort(tmp_path: Path, _patched_cd: None) -> None:
    bundle = load_coderdata_tranche("test_synthetic", tmp_path, ingestion_date=date(2026, 5, 28))

    # Matched cohort: P01, P02, P03 (P04 excluded for lacking RNA)
    assert bundle.tranche.patient_count == 3
    assert bundle.tranche.sample_count == 6  # 3 patients x 2 specimens each
    assert {p.patient_id for p in bundle.patients} == {"P01", "P02", "P03"}
    assert set(bundle.tranche.subtypes) == {"Osteosarcoma", "Ewing Sarcoma", "Leiomyosarcoma"}
    assert len(bundle.tranche.content_hash) == 64


def test_no_p04_in_drug_assays(tmp_path: Path, _patched_cd: None) -> None:
    bundle = load_coderdata_tranche("test_synthetic", tmp_path)
    # Cohort filter excludes P04, so no DrugAssay should reference its samples
    p04_sample_ids = {"P04_Tumor"}
    assert not any(a.sample_id in p04_sample_ids for a in bundle.drug_assays)


def test_response_metric_mapping(tmp_path: Path, _patched_cd: None) -> None:
    bundle = load_coderdata_tranche("test_synthetic", tmp_path)
    metrics = {a.response_metric for a in bundle.drug_assays}
    # published_auc -> auc, fit_ic50 -> ic50; fit_hs is skipped
    assert metrics == {"auc", "ic50"}
    # 3 AUC rows (P01x2 + P02x1) + 1 IC50 (P03)
    assert sum(1 for a in bundle.drug_assays if a.response_metric == "auc") == 3
    assert sum(1 for a in bundle.drug_assays if a.response_metric == "ic50") == 1
    # fit_hs entry was skipped
    assert len(bundle.drug_assays) == 4


def test_drug_response_attaches_to_tumor_sample(tmp_path: Path, _patched_cd: None) -> None:
    """CoderData puts drug response on Tumor samples; the adapter must preserve this."""
    bundle = load_coderdata_tranche("test_synthetic", tmp_path)
    for assay in bundle.drug_assays:
        assert assay.sample_id.endswith("_Tumor"), (
            f"assay {assay.assay_id} attached to non-Tumor sample {assay.sample_id}"
        )


def test_pubchem_resolution(tmp_path: Path, _patched_cd: None) -> None:
    bundle = load_coderdata_tranche("test_synthetic", tmp_path)
    by_drug = {a.drug_id: a for a in bundle.drug_assays}
    assert by_drug["D01"].pubchem_cid == 5291
    assert by_drug["D01"].inchikey == "INCHI-IMA-X"
    assert by_drug["D01"].drug_name == "imatinib"
    assert by_drug["D02"].pubchem_cid == 60700
    # D03 wasn't screened in the matched cohort; only P03xD03 was, which used
    # published_auc on sample 1003 (a Tumor)
    assert by_drug["D03"].pubchem_cid is None
    assert by_drug["D03"].inchikey is None


def test_baseline_expression_per_rna_sample(tmp_path: Path, _patched_cd: None) -> None:
    bundle = load_coderdata_tranche("test_synthetic", tmp_path)
    # All 6 cohort samples have RNA (Tumor + Organoid for each of P01-P03)
    assert len(bundle.baseline_expression) == 6
    for be in bundle.baseline_expression:
        assert be.normalization == "tpm"
        assert be.gene_id_namespace == "entrez"
        assert be.gene_count == 3


def test_expression_anndata_shape(tmp_path: Path, _patched_cd: None) -> None:
    bundle = load_coderdata_tranche("test_synthetic", tmp_path)
    assert bundle.expression.shape == (6, 3)
    assert "patient_id" in bundle.expression.obs.columns
    assert "model_type" in bundle.expression.obs.columns


def test_cancer_type_filter(tmp_path: Path, _patched_cd: None) -> None:
    bundle = load_coderdata_tranche("test_synthetic", tmp_path, cancer_type_filter=["Osteosarcoma"])
    # Filter to Osteosarcoma -> only P01 in cohort
    assert bundle.tranche.patient_count == 1
    assert {p.patient_id for p in bundle.patients} == {"P01"}
    assert bundle.tranche.tranche_id.startswith("coderdata_test_synthetic__osteosarcoma")


def test_cancer_type_filter_empty_raises(tmp_path: Path, _patched_cd: None) -> None:
    with pytest.raises(IngestError, match="no patients have BOTH"):
        load_coderdata_tranche(
            "test_synthetic",
            tmp_path,
            cancer_type_filter=["NotACancerType"],
        )


def test_content_hash_deterministic(tmp_path: Path, _patched_cd: None) -> None:
    a = load_coderdata_tranche("test_synthetic", tmp_path).tranche.content_hash
    b = load_coderdata_tranche("test_synthetic", tmp_path).tranche.content_hash
    assert a == b
