"""Loader for the Soragni 2024 PDTO sarcoma tranche.

End-to-end pipeline:

1. Verify the raw artifact manifest (``data/raw/soragni/tables/manifest.json``)
   against on-disk shas; refuse to proceed on mismatch.
2. Parse normalized-gene-counts sample columns (e.g. ``SARC0095_Tumor``,
   ``SARC0139_1_Organoids``) into ``(patient_id, specimen)`` pairs, ignoring
   the two trailing ``col`` / ``col1`` empty straggler columns.
3. Canonicalize drug-screen ``Sample_ID`` (lower-cased ``sarcNNNN``) to the
   same ``SARC<digits>[_<digits>]`` form, then intersect with the RNA-seq
   patient set -- this is the matched drug x RNA cohort.
4. Build an AnnData (cells x genes; ``X`` = pre-computed median-of-ratios
   normalized counts from the Soragni 2024 protocol; ``var`` indexed by
   Ensembl gene_id with Gene_Symbol + biotype as columns).
5. Resolve every drug name through ``fmharness.data.drug_xref`` to attach
   PubChem CID / InChIKey / DrugBank ID.
6. Emit Pydantic schema objects: one ``Patient`` per matched patient, one
   ``Sample`` + ``BaselineExpression`` per (patient, specimen) pair, and
   one ``DrugAssay`` per (organoid-sample, drug) -- Soragni screens drugs
   on the organoids, so dose-response attaches to the Organoid sample_id.
   The Tumor sample is retained for baseline-comparison / sensitivity rows.
7. Wrap everything in a content-hashed ``Tranche``.

Returned bundle is in-memory. Persistence (writing the AnnData to
``data/tranches/...``) is a separate step the CLI will own.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.data.drug_xref import load_drug_xref
from fmharness.schema import (
    BaselineExpression,
    DrugAssay,
    Patient,
    Sample,
    Tranche,
)

TRANCHE_ID = "soragni_pdo_sarcoma_2024"
SOURCE = "soragni_pdo_sarcoma"
VERSION = "syn55180195_tables"
TISSUE_OF_ORIGIN = "sarcoma"

NGC_GENE_META_COLS = (
    "Chromosome",
    "Start",
    "Stop",
    "Length",
    "Strand",
    "Gene_Symbol",
    "gene_biotype",
    "gene_id",
    "gene_name",
    "gene_source",
    "gene_version",
)

# Pattern matches a Soragni sample identifier:
#   SARC<digits>                          -> patient only (e.g. "sarc0001")
#   SARC<digits>_<alphanumeric>           -> patient with a timepoint or anatomical suffix
#                                            (e.g. "SARC0139_1", "sarc0024_B", "sarc0028_biopsy")
#   SARC<digits>[_<alnum>]_Tumor|Organoid(s)? -> patient + specimen marker (RNA-seq column form)
# The negative lookahead on the inner group keeps the specimen vocabulary
# from being captured as part of the patient ID.
_SORAGNI_ID_RE = re.compile(
    r"^(?P<patient>SARC\d+(?:_(?!Tumor$|Organoids?$)[A-Za-z0-9]+)?)"
    r"(?:_(?P<specimen>Tumor|Organoids?))?$",
    re.IGNORECASE,
)


class IngestError(RuntimeError):
    """Raised when raw artifact verification or join logic fails."""


@dataclass(frozen=True)
class SoragniBundle:
    """In-memory result of ``load_soragni()``.

    ``expression`` is an AnnData with cells along ``obs`` (one per
    ``(patient, specimen)`` pair; both Tumor and Organoid retained) and
    genes along ``var`` (Ensembl ``gene_id`` as index, ``Gene_Symbol`` +
    ``gene_biotype`` as columns). ``X`` is the Soragni-provided
    median-of-ratios normalized matrix.
    """

    tranche: Tranche
    patients: list[Patient]
    samples: list[Sample]
    expression: ad.AnnData
    drug_assays: list[DrugAssay]
    baseline_expression: list[BaselineExpression]


def canonicalize_patient_id(s: str) -> str:
    """Normalize any Soragni sample identifier to its canonical patient form.

    Drops a trailing ``_Tumor`` / ``_Organoid(s)`` suffix and uppercases.

        >>> canonicalize_patient_id("sarc0001")
        'SARC0001'
        >>> canonicalize_patient_id("SARC0139_1")
        'SARC0139_1'
        >>> canonicalize_patient_id("SARC0065_Tumor")
        'SARC0065'
    """
    m = _SORAGNI_ID_RE.match(s)
    if not m:
        raise ValueError(f"unrecognized Soragni sample ID: {s!r}")
    return m.group("patient").upper()


def load_soragni(
    repo_root: Path,
    *,
    ingestion_date: date | None = None,
    verify_manifest: bool = True,
) -> SoragniBundle:
    """Load the Soragni PDTO sarcoma tranche end-to-end (see module docstring)."""
    raw_dir = repo_root / "data" / "raw" / "soragni" / "tables"
    if verify_manifest:
        _verify_raw_manifest(raw_dir)

    ngc = pd.read_parquet(raw_dir / "normalized_gene_counts.parquet")
    drug_screen = pd.read_parquet(raw_dir / "drug_screen.parquet")

    # ---- Step 2: parse sample columns ----
    # Anything that's not gene metadata and matches the SARC pattern.
    column_to_patient_specimen: dict[str, tuple[str, str]] = {}
    for col in ngc.columns:
        if col in NGC_GENE_META_COLS:
            continue
        m = _SORAGNI_ID_RE.match(col)
        if not m or not m.group("specimen"):
            # silently drop stragglers like "col" / "col1" or any non-conforming column
            continue
        patient = m.group("patient").upper()
        specimen = m.group("specimen").lower().rstrip("s")  # Tumor or Organoid
        column_to_patient_specimen[col] = (patient, specimen.capitalize())

    # ---- Step 3: matched cohort ----
    ngc_patients = {pat for pat, _ in column_to_patient_specimen.values()}
    drug_screen = drug_screen.copy()
    drug_screen["_patient"] = drug_screen["Sample_ID"].apply(canonicalize_patient_id)
    drug_patients = set(drug_screen["_patient"])
    matched = sorted(ngc_patients & drug_patients)
    if not matched:
        raise IngestError("no overlap between RNA-seq patients and drug-screen patients")

    cohort_samples = sorted(
        [
            (col, pat, spec)
            for col, (pat, spec) in column_to_patient_specimen.items()
            if pat in matched
        ]
    )

    # ---- Step 4: AnnData ----
    sample_ids = [f"{pat}_{spec}" for _, pat, spec in cohort_samples]
    cohort_cols = [col for col, _, _ in cohort_samples]

    # ngc[cohort_cols] is genes x samples; transpose to samples x genes
    X = ngc[cohort_cols].to_numpy(dtype=np.float64).T

    # var: use gene_id (Ensembl) as the index. Keep Gene_Symbol + gene_biotype.
    var = ngc[list(NGC_GENE_META_COLS)].copy()
    var = var.set_index("gene_id")
    # Drop columns the harness does not need downstream; keep symbol + biotype + chromosome.
    var = var[["Gene_Symbol", "gene_biotype", "gene_name", "Chromosome"]]

    obs = pd.DataFrame(
        {
            "patient_id": [pat for _, pat, _ in cohort_samples],
            "specimen": [spec for _, _, spec in cohort_samples],
            "ngc_column": cohort_cols,
        },
        index=sample_ids,
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["normalization"] = "median_of_ratios"
    adata.uns["source"] = SOURCE
    adata.uns["version"] = VERSION

    # ---- Step 5: drug xref ----
    xref = load_drug_xref(repo_root / "data" / "static")
    soragni_xref = xref[xref["source"] == "soragni"]
    soragni_xref_unique = soragni_xref.drop_duplicates(subset=["input_name"], keep="first")
    name_to_xref = soragni_xref_unique.set_index(soragni_xref_unique["input_name"].str.lower())

    # ---- Step 6: schema objects ----
    # Per-patient diagnosis (Soragni's free-text Diagnosis column from drug_screen).
    diag_by_patient = (
        drug_screen.drop_duplicates(subset=["_patient"])
        .set_index("_patient")["Diagnosis"]
        .to_dict()
    )

    cohort_subtypes: set[str] = set()
    patients: list[Patient] = []
    for pat in matched:
        diag = diag_by_patient.get(pat)
        if diag is not None and pd.notna(diag):
            cohort_subtypes.add(str(diag))
        patients.append(
            Patient(
                patient_id=pat,
                tranche_id=TRANCHE_ID,
                tissue_of_origin=TISSUE_OF_ORIGIN,
                subtype=str(diag) if diag is not None and pd.notna(diag) else None,
                subtype_granularity="fine",
            )
        )

    samples: list[Sample] = []
    baseline_expression: list[BaselineExpression] = []
    # Map patient -> Organoid sample_id (for drug-assay attachment)
    patient_to_organoid_sample: dict[str, str] = {}
    for sample_id, (col, pat, spec) in zip(sample_ids, cohort_samples, strict=False):
        samples.append(
            Sample(
                sample_id=sample_id,
                patient_id=pat,
                tranche_id=TRANCHE_ID,
                metadata={"specimen": spec, "ngc_column": col},
            )
        )
        baseline_expression.append(
            BaselineExpression(
                sample_id=sample_id,
                expression_matrix_uri=f"tranche://{TRANCHE_ID}/expression.h5ad#obs/{sample_id}",
                gene_count=adata.shape[1],
                gene_id_namespace="ensembl",
                normalization="median_of_ratios",
                reference_genome="GRCh38",
                reference_annotation="GENCODE",  # Havana/Ensembl_Havana per gene_source column
            )
        )
        if spec == "Organoid":
            patient_to_organoid_sample[pat] = sample_id

    # ---- DrugAssay rows ----
    # Soragni screens drugs on the organoids; attach response to Organoid sample_id.
    cohort_drug_rows = drug_screen[drug_screen["_patient"].isin(matched)]
    drug_assays: list[DrugAssay] = []
    for _, row in cohort_drug_rows.iterrows():
        pat = row["_patient"]
        organoid_sample = patient_to_organoid_sample.get(pat)
        if organoid_sample is None:
            # Patient is matched (has RNA-seq) but no Organoid specimen specifically.
            # Defensive: every Soragni RNA-sampled patient has both Tumor and Organoid
            # in syn64333318 today, but the schema does not guarantee that, so skip safely.
            continue
        drug_name = str(row["Drug_Name"])
        xref_row = (
            name_to_xref.loc[drug_name.lower()] if drug_name.lower() in name_to_xref.index else None
        )
        cid = _maybe_int(xref_row.get("pubchem_cid")) if xref_row is not None else None
        inchikey = _maybe_str(xref_row.get("inchikey")) if xref_row is not None else None
        drugbank = _maybe_str(xref_row.get("drugbank_id")) if xref_row is not None else None

        # Soragni has no separate DRUG_ID column; use the drug name as drug_id too.
        # Downstream code joins to drug_xref on pubchem_cid for canonicalization.
        drug_assays.append(
            DrugAssay(
                assay_id=f"{organoid_sample}__{drug_name}",
                sample_id=organoid_sample,
                drug_id=drug_name,
                drug_name=drug_name,
                response_metric="viability",
                response_value=float(row["Viability_Score"]),
                pubchem_cid=cid,
                inchikey=inchikey,
                drugbank_id=drugbank,
            )
        )

    # ---- Step 7: Tranche metadata ----
    tranche = Tranche(
        tranche_id=TRANCHE_ID,
        source=SOURCE,
        version=VERSION,
        ingestion_date=ingestion_date or date.today(),
        patient_count=len(patients),
        sample_count=len(samples),
        drug_count=len({a.drug_id for a in drug_assays}),
        subtypes=tuple(sorted(cohort_subtypes)),
        content_hash=_content_hash(raw_dir),
        description=(
            "Soragni 2024 PDTO sarcoma biobank: pre-computed normalized gene counts "
            "(syn64333318) + organoid drug screen (Viability_Score, % of vehicle "
            "control). Drug response attaches to the Organoid sample_id."
        ),
    )

    return SoragniBundle(
        tranche=tranche,
        patients=patients,
        samples=samples,
        expression=adata,
        drug_assays=drug_assays,
        baseline_expression=baseline_expression,
    )


def _verify_raw_manifest(raw_dir: Path) -> None:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise IngestError(
            f"raw manifest missing: {manifest_path} -- run "
            "scripts/download/download_soragni.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    for rel, rec in manifest.get("files", {}).items():
        path = raw_dir / rel
        if not path.exists():
            raise IngestError(f"raw file missing: {rel}")
        actual = _sha256(path)
        if actual != rec["sha256"]:
            raise IngestError(f"sha256 mismatch on {rel}: expected {rec['sha256']}, got {actual}")


def _content_hash(raw_dir: Path) -> str:
    """sha256 of (VERSION + sorted-rel-paths + their shas). Deterministic per input set."""
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    h = hashlib.sha256()
    h.update(VERSION.encode())
    for rel in sorted(manifest["files"]):
        h.update(rel.encode())
        h.update(manifest["files"][rel]["sha256"].encode())
    return h.hexdigest()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _maybe_str(v: object) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def _maybe_int(v: object) -> int | None:
    if v is None or pd.isna(v):
        return None
    return int(v)
