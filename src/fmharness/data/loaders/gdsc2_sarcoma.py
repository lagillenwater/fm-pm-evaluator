"""Loader for the GDSC2 sarcoma tranche.

End-to-end pipeline:

1. Verify the raw artifact manifest (``data/raw/gdsc2_sarcoma/manifest.json``)
   against on-disk shas; refuse to proceed on mismatch.
2. Filter DepMap ``Model.csv`` to ``OncotreeLineage`` in ``{"Soft Tissue",
   "Bone"}``; keep only models with a populated ``COSMICID`` (the join key
   into GDSC2 dose-response).
3. Filter GDSC2 ``GDSC2_fitted_dose_response_27Oct23.xlsx`` to the sarcoma
   COSMIC set; collect the cohort = lines present in *both* DepMap Model.csv
   AND GDSC2 dose-response AND DepMap RNA-seq.
4. Slice the DepMap RNA-seq matrix
   (``OmicsExpressionRawReadCountHumanProteinCodingGenes.csv``) to the
   cohort ACH IDs (using ``IsDefaultEntryForModel == "Yes"`` to dedupe).
   Parse gene-column headers (``"SYMBOL (ENTREZ_ID)"``) into ``var``.
5. Run pydeseq2 ``fit_size_factors`` -> median-of-ratios normalization, so
   the GDSC2 expression matrix is normalized by the same method as the
   Soragni pre-computed counts (closing the upstream-pipeline confounder).
6. Resolve every drug name through ``fmharness.data.drug_xref`` to attach
   PubChem CID / InChIKey / DrugBank ID. Unresolvable drugs keep the row
   but leave the xref fields null.
7. Emit Pydantic schema objects: one ``Patient`` and one ``Sample`` per
   ACH ID, one ``BaselineExpression`` per ACH ID, two ``DrugAssay`` rows
   per (cell, GDSC2 drug entry) -- one for IC50, one for AUC.
8. Wrap everything in a content-hashed ``Tranche``.

Returned bundle is in-memory. Persistence (writing the AnnData to
``data/tranches/...``) is a separate step the CLI will own on Day 5.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet

from fmharness.data._pandas_utils import maybe_int, maybe_str
from fmharness.data.drug_xref import load_drug_xref
from fmharness.schema import (
    BaselineExpression,
    DrugAssay,
    Patient,
    Sample,
    Tranche,
)

TRANCHE_ID = "gdsc2_sarcoma_release8.5"
SOURCE = "gdsc2_sarcoma"
VERSION = "release8.5_depmap26q1"
SARCOMA_LINEAGES = ("Soft Tissue", "Bone")

_DEPMAP_METADATA_COLS = (
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
)

# Gene column headers in DepMap are "HGNC_SYMBOL (ENTREZ_ID)" -- e.g. "TSPAN6 (7105)".
_GENE_HEADER_RE = re.compile(r"^(?P<symbol>.+?)\s*\((?P<entrez>\d+)\)\s*$")


class IngestError(RuntimeError):
    """Raised when raw artifact verification or join logic fails."""


@dataclass(frozen=True)
class GDSC2SarcomaBundle:
    """In-memory result of ``load_gdsc2_sarcoma()``.

    The ``expression`` AnnData has cells along ``obs`` (one per ACH ID) and
    genes along ``var`` (HGNC symbols as index, Entrez ID as a ``var``
    column). ``X`` is the DESeq2 median-of-ratios normalized matrix
    (``float64``); ``layers["raw_counts"]`` is the original integer matrix.
    """

    tranche: Tranche
    patients: list[Patient]
    samples: list[Sample]
    expression: ad.AnnData
    drug_assays: list[DrugAssay]
    baseline_expression: list[BaselineExpression]


def load_gdsc2_sarcoma(
    repo_root: Path,
    *,
    ingestion_date: date | None = None,
    verify_manifest: bool = True,
) -> GDSC2SarcomaBundle:
    """Load the GDSC2 sarcoma tranche end-to-end (see module docstring)."""
    raw_dir = repo_root / "data" / "raw" / "gdsc2_sarcoma"
    if verify_manifest:
        _verify_raw_manifest(raw_dir)

    model_path = raw_dir / "depmap" / "Model.csv"
    expr_path = raw_dir / "depmap" / "OmicsExpressionRawReadCountHumanProteinCodingGenes.csv"
    gdsc_path = raw_dir / "gdsc2" / "GDSC2_fitted_dose_response_27Oct23.xlsx"

    # ---- Step 2: sarcoma model filter ----
    model_df = pd.read_csv(model_path, low_memory=False)
    sarcoma_models = cast(
        pd.DataFrame, model_df[model_df["OncotreeLineage"].isin(list(SARCOMA_LINEAGES))].copy()
    )
    sarcoma_models = sarcoma_models.dropna(subset=["COSMICID"])
    sarcoma_models["COSMICID"] = sarcoma_models["COSMICID"].astype(int)

    # ---- Step 3: GDSC2 dose-response filter ----
    gdsc_df = pd.read_excel(gdsc_path, sheet_name=0)
    sarcoma_cosmic = list(sarcoma_models["COSMICID"])
    gdsc_sarcoma = cast(pd.DataFrame, gdsc_df[gdsc_df["COSMIC_ID"].isin(sarcoma_cosmic)].copy())
    cohort_cosmic = list(gdsc_sarcoma["COSMIC_ID"].astype(int).unique())
    cohort_models = cast(
        pd.DataFrame, sarcoma_models[sarcoma_models["COSMICID"].isin(cohort_cosmic)].copy()
    )

    # ---- Step 4: slice expression matrix to cohort ACH IDs ----
    # The CSV has a leading unnamed numeric index column; absorb it via index_col=0.
    expr_raw = pd.read_csv(expr_path, index_col=0)
    # Dedupe to one row per ModelID (DepMap publishes multiple sequencing reps; one is flagged
    # as the canonical entry for the Model).
    expr_default = cast(pd.DataFrame, expr_raw[expr_raw["IsDefaultEntryForModel"] == "Yes"].copy())
    expr_default = expr_default.set_index("ModelID")
    gene_cols = [c for c in expr_default.columns if c not in _DEPMAP_METADATA_COLS]

    cohort_ach = sorted(set(cohort_models["ModelID"]) & set(expr_default.index))
    if not cohort_ach:
        raise IngestError(
            "no overlap between sarcoma cohort (from Model.csv+GDSC2) and DepMap RNA-seq"
        )
    expr_cohort = expr_default.loc[cohort_ach, gene_cols].astype(int)
    cohort_models = cast(
        pd.DataFrame, cohort_models[cohort_models["ModelID"].isin(cohort_ach)].copy()
    )
    cohort_models = cohort_models.set_index("ModelID").loc[cohort_ach]

    # ---- Step 5: pydeseq2 median-of-ratios ----
    counts_for_dds = expr_cohort.astype("int64")
    dds_metadata = pd.DataFrame(
        {"condition": ["A"] * len(counts_for_dds)}, index=counts_for_dds.index
    )
    dds = DeseqDataSet(
        counts=counts_for_dds,
        metadata=dds_metadata,
        design="~1",
        quiet=True,
    )
    dds.fit_size_factors()
    normed = np.asarray(dds.layers["normed_counts"], dtype=np.float64)
    size_factors = np.asarray(dds.obs["size_factors"].values, dtype=np.float64)

    # ---- Build AnnData ----
    var_records: list[dict[str, object]] = []
    for col in gene_cols:
        m = _GENE_HEADER_RE.match(col)
        if m:
            var_records.append({"symbol": m["symbol"], "entrez": int(m["entrez"])})
        else:
            var_records.append({"symbol": col, "entrez": -1})
    var = pd.DataFrame(var_records)
    # Disambiguate any duplicate symbols by suffixing the Entrez ID
    if var["symbol"].duplicated().any():
        dup_mask = var["symbol"].duplicated(keep=False)
        var.loc[dup_mask, "symbol"] = (
            var.loc[dup_mask, "symbol"] + "__" + var.loc[dup_mask, "entrez"].astype(str)
        )
    var = var.set_index("symbol")

    obs = cohort_models[["OncotreeLineage", "OncotreeSubtype", "COSMICID"]].copy()
    obs["size_factor"] = size_factors

    adata = ad.AnnData(X=normed, obs=obs, var=var)
    adata.layers["raw_counts"] = expr_cohort.values.astype(np.int64)
    adata.uns["normalization"] = "median_of_ratios"
    adata.uns["source"] = SOURCE
    adata.uns["version"] = VERSION

    # ---- Step 6: drug xref ----
    xref = load_drug_xref(repo_root / "data" / "static")
    gdsc_xref = cast(pd.DataFrame, xref[xref["source"] == "gdsc2"])
    # Map drug_name (lowercased) -> first row of xref metadata
    gdsc_xref_unique = gdsc_xref.drop_duplicates(subset=["input_name"], keep="first")
    name_to_xref = gdsc_xref_unique.set_index(gdsc_xref_unique["input_name"].str.lower())

    # ---- Step 7: build schema objects ----
    patients: list[Patient] = []
    samples: list[Sample] = []
    baseline_expression: list[BaselineExpression] = []
    cohort_subtypes: set[str] = set()
    for ach_id, row in cohort_models.iterrows():
        subtype = row["OncotreeSubtype"] if pd.notna(row["OncotreeSubtype"]) else None
        if subtype is not None:
            cohort_subtypes.add(subtype)
        patients.append(
            Patient(
                patient_id=str(ach_id),
                tranche_id=TRANCHE_ID,
                tissue_of_origin=row["OncotreeLineage"],
                subtype=subtype,
                subtype_granularity="fine",
                metadata={
                    "cosmic_id": int(row["COSMICID"]),
                    "ccle_name": maybe_str(row.get("CCLEName")),
                    "sanger_model_id": maybe_str(row.get("SangerModelID")),
                },
            )
        )
        samples.append(Sample(sample_id=str(ach_id), patient_id=str(ach_id), tranche_id=TRANCHE_ID))
        baseline_expression.append(
            BaselineExpression(
                sample_id=str(ach_id),
                expression_matrix_uri=f"tranche://{TRANCHE_ID}/expression.h5ad#obs/{ach_id}",
                gene_count=adata.shape[1],
                gene_id_namespace="symbol",
                normalization="median_of_ratios",
                reference_genome="GRCh38",
                reference_annotation=None,
            )
        )

    # ---- DrugAssay rows (one IC50 + one AUC per (cell, drug)) ----
    cosmic_to_ach = dict(zip(cohort_models["COSMICID"], cohort_models.index, strict=False))
    gdsc_sarcoma = gdsc_sarcoma[gdsc_sarcoma["COSMIC_ID"].astype(int).isin(cosmic_to_ach)].copy()

    drug_assays: list[DrugAssay] = []
    for _, row in gdsc_sarcoma.iterrows():
        ach_id = cosmic_to_ach[int(row["COSMIC_ID"])]
        drug_id = str(row["DRUG_ID"])
        drug_name = str(row["DRUG_NAME"])
        xref_row = (
            name_to_xref.loc[drug_name.lower()] if drug_name.lower() in name_to_xref.index else None
        )
        cid = maybe_int(xref_row.get("pubchem_cid")) if xref_row is not None else None
        inchikey = maybe_str(xref_row.get("inchikey")) if xref_row is not None else None
        drugbank = maybe_str(xref_row.get("drugbank_id")) if xref_row is not None else None

        drug_assays.append(
            DrugAssay(
                assay_id=f"{ach_id}__{drug_id}__ic50",
                sample_id=str(ach_id),
                drug_id=drug_id,
                drug_name=drug_name,
                response_metric="ic50",
                response_value=float(row["LN_IC50"]),
                pubchem_cid=cid,
                inchikey=inchikey,
                drugbank_id=drugbank,
            )
        )
        drug_assays.append(
            DrugAssay(
                assay_id=f"{ach_id}__{drug_id}__auc",
                sample_id=str(ach_id),
                drug_id=drug_id,
                drug_name=drug_name,
                response_metric="auc",
                response_value=float(row["AUC"]),
                pubchem_cid=cid,
                inchikey=inchikey,
                drugbank_id=drugbank,
            )
        )

    # ---- Step 8: Tranche metadata ----
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
            "GDSC2 dose-response paired with DepMap RNA-seq (raw counts -> DESeq2 "
            "median-of-ratios), filtered to sarcoma cell lines."
        ),
    )

    return GDSC2SarcomaBundle(
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
            "scripts/download/download_gdsc2_sarcoma.py first"
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
