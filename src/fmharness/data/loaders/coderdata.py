"""Loader that turns a CoderData dataset into our Pydantic tranche objects.

CoderData (PNNL) ships ~18 datasets in a uniform LinkML-defined schema:
``samples``, ``transcriptomics`` (long-format TPM), ``experiments`` (drug
response), ``drugs`` (PubChem CID + InChIKey + SMILES), ``drug_descriptors``
(Morgan fingerprints), and others. Because the schema is identical across
datasets, this single loader replaces the per-dataset loaders we used to
maintain.

Conventions adopted here:

- Drug response stays attached to whichever sample CoderData put it on.
  For ``sarcoma`` that's the Tumor sample; for ``gdscv2`` it's the cell-line
  sample. We do not move it.
- Patient IDs come verbatim from CoderData's ``common_name`` field. CoderData
  uses mixed conventions (e.g. ``SARC0069-2`` with a hyphen alongside
  ``SARC0139_1`` with an underscore); we preserve those rather than
  normalizing.
- Matched cohort is patient-level: a patient is included if they have at
  least one sample with transcriptomics AND at least one sample with
  experiments. The two need not be the same sample.
- Expression normalization is TPM (CoderData's choice). Same processing
  pipeline across all CoderData datasets, which is the cross-cohort
  comparability we want for substrate-gap analyses.

Map of CoderData ``dose_response_metric`` -> our ``ResponseMetric``:
``published_auc`` / ``fit_auc`` / ``auc`` -> ``"auc"``, ``fit_ic50`` ->
``"ic50"``, ``aac`` -> ``"aac"``, ``dss`` -> ``"dss"``. Curve-fit
parameters (``fit_ec50``, ``fit_r2``, ``fit_ec50se``, ``fit_einf``,
``fit_hs``) are skipped at MVP scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import anndata as ad
import coderdata as cd
import numpy as np
import pandas as pd

from fmharness.data._pandas_utils import maybe_int, maybe_str
from fmharness.schema import (
    BaselineExpression,
    DrugAssay,
    Patient,
    ResponseMetric,
    Sample,
    Tranche,
)

CODERDATA_LOCAL_PATH_DEFAULT = "data/raw/coderdata"

# Map CoderData's dose_response_metric strings to our ResponseMetric literal.
# Anything not in this dict is skipped at MVP scope.
_RESPONSE_METRIC_MAP: dict[str, str] = {
    "published_auc": "auc",
    "fit_auc": "auc",
    "auc": "auc",
    "fit_ic50": "ic50",
    "aac": "aac",
    "dss": "dss",
}


class IngestError(RuntimeError):
    """Raised when CoderData inputs are malformed or unusable."""


@dataclass(frozen=True)
class CoderDataBundle:
    """In-memory tranche produced by ``load_coderdata_tranche``.

    ``expression`` is an AnnData with samples on ``obs`` (one per sample
    with transcriptomics in the matched cohort) and genes on ``var``
    (Entrez ID as index). ``X`` is TPM (CoderData's normalization).
    """

    tranche: Tranche
    patients: list[Patient]
    samples: list[Sample]
    expression: ad.AnnData
    drug_assays: list[DrugAssay]
    baseline_expression: list[BaselineExpression]


def load_coderdata_tranche(
    name: str,
    repo_root: Path,
    *,
    cancer_type_filter: list[str] | None = None,
    ingestion_date: date | None = None,
    local_path: str = CODERDATA_LOCAL_PATH_DEFAULT,
) -> CoderDataBundle:
    """Load one CoderData dataset into a tranche of Pydantic schema objects.

    ``name`` is a CoderData dataset key (``"sarcoma"``, ``"gdscv2"``,
    ``"liver"``, etc. -- see ``coderdata.list_datasets()``).
    ``cancer_type_filter`` optionally restricts to a list of CoderData
    ``cancer_type`` strings (used to slice ``gdscv2`` to sarcoma lines).
    Files are downloaded under ``repo_root / local_path`` if not already
    present.
    """
    ds_path = repo_root / local_path
    ds_path.mkdir(parents=True, exist_ok=True)
    cd.download(name=name, local_path=ds_path, exist_ok=True)
    ds = cd.load(name, local_path=ds_path)

    samples_df: pd.DataFrame = ds.samples
    transcriptomics_df: pd.DataFrame = ds.transcriptomics
    experiments_df: pd.DataFrame = ds.experiments
    drugs_df: pd.DataFrame = ds.drugs

    for tname, val in [
        ("samples", samples_df),
        ("transcriptomics", transcriptomics_df),
        ("experiments", experiments_df),
        ("drugs", drugs_df),
    ]:
        if val is None or len(val) == 0:
            raise IngestError(f"CoderData dataset {name!r}: missing {tname} table")

    # ---- cancer-type filter (for gdscv2 sarcoma slice and similar) ----
    if cancer_type_filter is not None:
        samples_df = cast(
            pd.DataFrame, samples_df[samples_df["cancer_type"].isin(cancer_type_filter)]
        )

    # ---- patient-level matched cohort ----
    rna_sample_ids = list(transcriptomics_df["improve_sample_id"].unique())
    exp_sample_ids = list(experiments_df["improve_sample_id"].unique())
    rna_sample_set = set(rna_sample_ids)
    exp_sample_set = set(exp_sample_ids)
    rna_patients = {
        str(p)
        for p in cast(
            pd.Series,
            samples_df[samples_df["improve_sample_id"].isin(rna_sample_ids)]["common_name"],
        )
        .dropna()
        .unique()
    }
    exp_patients = {
        str(p)
        for p in cast(
            pd.Series,
            samples_df[samples_df["improve_sample_id"].isin(exp_sample_ids)]["common_name"],
        )
        .dropna()
        .unique()
    }
    matched_patients = sorted(rna_patients & exp_patients)
    if not matched_patients:
        raise IngestError(
            f"CoderData dataset {name!r}: no patients have BOTH transcriptomics and experiments"
            + (f" within cancer_type_filter={cancer_type_filter}" if cancer_type_filter else "")
        )

    # ---- filter all three tables to matched-cohort patients ----
    cohort_samples = cast(
        pd.DataFrame, samples_df[samples_df["common_name"].isin(matched_patients)].copy()
    )
    cohort_sample_ids = set(cohort_samples["improve_sample_id"])

    rna_cohort_ids = sorted(cohort_sample_ids & rna_sample_set)
    exp_cohort_ids = list(cohort_sample_ids & exp_sample_set)

    transcriptomics_cohort = cast(
        pd.DataFrame,
        transcriptomics_df[transcriptomics_df["improve_sample_id"].isin(rna_cohort_ids)],
    )
    experiments_cohort = cast(
        pd.DataFrame,
        experiments_df[experiments_df["improve_sample_id"].isin(exp_cohort_ids)],
    )

    # ---- AnnData: pivot long-format transcriptomics to (sample x gene) ----
    expr_wide = transcriptomics_cohort.pivot_table(
        index="improve_sample_id",
        columns="entrez_id",
        values="transcriptomics",
        aggfunc="mean",  # in case of duplicate (sample, gene) rows
    )
    # Order rows to match sorted sample IDs; var index is the gene Entrez IDs
    expr_wide = expr_wide.loc[rna_cohort_ids]
    expr_wide.columns = expr_wide.columns.astype(int)
    var = pd.DataFrame(index=expr_wide.columns.astype(str), data={"entrez_id": expr_wide.columns})
    var.index.name = "entrez_id_str"

    # obs ordered to expr_wide; carry patient_id, sample_other_id, model_type, cancer_type
    sample_info_by_id = cohort_samples.drop_duplicates(subset=["improve_sample_id"]).set_index(
        "improve_sample_id"
    )
    obs_rows = sample_info_by_id.loc[rna_cohort_ids][
        ["other_id", "common_name", "model_type", "cancer_type"]
    ].rename(columns={"common_name": "patient_id"})
    obs_rows.index = obs_rows.index.astype(str)

    X = expr_wide.to_numpy(dtype=np.float64)
    adata = ad.AnnData(X=X, obs=obs_rows, var=var)
    adata.uns["normalization"] = "tpm"
    adata.uns["source"] = f"coderdata_{name}"
    adata.uns["coderdata_version"] = cd.__version__

    # ---- schema objects ----
    # subtype per patient: use cancer_type from any cohort_sample row for that patient
    patient_subtype: dict[str, str | None] = {}
    for pat in matched_patients:
        ct_vals = cohort_samples.loc[cohort_samples["common_name"] == pat, "cancer_type"]
        non_null = [str(v) for v in ct_vals if pd.notna(v)]
        patient_subtype[pat] = non_null[0] if non_null else None

    tranche_id = f"coderdata_{name}"
    if cancer_type_filter is not None:
        # Tag the tranche with a short suffix so two slices of the same dataset
        # (e.g. all of gdscv2 vs. its sarcoma slice) don't collide.
        slug = "_".join(sorted(c.lower().replace(" ", "_") for c in cancer_type_filter))[:32]
        tranche_id = f"{tranche_id}__{slug}"

    patients: list[Patient] = []
    for pat in matched_patients:
        patients.append(
            Patient(
                patient_id=pat,
                tranche_id=tranche_id,
                tissue_of_origin="cancer",
                subtype=patient_subtype.get(pat),
                subtype_granularity="fine",
                metadata={},
            )
        )

    sample_objs: list[Sample] = []
    baseline_expression: list[BaselineExpression] = []
    for sample_id in cohort_sample_ids:
        srow = sample_info_by_id.loc[sample_id]
        other_id = str(srow["other_id"])
        common_name = str(srow["common_name"])
        model_type_str = maybe_str(srow.get("model_type"))
        sample_objs.append(
            Sample(
                sample_id=other_id,
                patient_id=common_name,
                tranche_id=tranche_id,
                metadata={
                    "improve_sample_id": int(sample_id),
                    "model_type": model_type_str,
                },
            )
        )
        # BaselineExpression only for samples that have transcriptomics
        if sample_id in rna_sample_ids:
            baseline_expression.append(
                BaselineExpression(
                    sample_id=other_id,
                    expression_matrix_uri=(
                        f"tranche://{tranche_id}/expression.h5ad#obs/{sample_id}"
                    ),
                    gene_count=adata.shape[1],
                    gene_id_namespace="entrez",
                    normalization="tpm",
                    reference_genome="GRCh38",
                    reference_annotation=None,
                )
            )

    # Drug-name + xref attachment: build a lookup from improve_drug_id -> first
    # row of drugs (in case of multiple synonyms per drug)
    drugs_unique = drugs_df.drop_duplicates(subset=["improve_drug_id"], keep="first").set_index(
        "improve_drug_id"
    )

    drug_assays: list[DrugAssay] = []
    skipped_metrics: dict[str, int] = {}
    for _, row in experiments_cohort.iterrows():
        raw_metric = str(row["dose_response_metric"])
        mapped_metric = _RESPONSE_METRIC_MAP.get(raw_metric)
        if mapped_metric is None:
            skipped_metrics[raw_metric] = skipped_metrics.get(raw_metric, 0) + 1
            continue
        improve_sid = int(row["improve_sample_id"])
        improve_did = str(row["improve_drug_id"])
        # Resolve to other_id (sample_id) and pull drug metadata
        sample_other_id = str(sample_info_by_id.loc[improve_sid]["other_id"])
        if improve_did in drugs_unique.index:
            drow = drugs_unique.loc[improve_did]
            drug_name = str(drow["chem_name"])
            pubchem = maybe_int(drow.get("pubchem_id"))
            inchikey = maybe_str(drow.get("InChIKey"))
        else:
            drug_name = improve_did
            pubchem = None
            inchikey = None
        drug_assays.append(
            DrugAssay(
                assay_id=f"{sample_other_id}__{improve_did}__{mapped_metric}",
                sample_id=sample_other_id,
                drug_id=improve_did,
                drug_name=drug_name,
                response_metric=cast(ResponseMetric, mapped_metric),
                response_value=float(row["dose_response_value"]),
                pubchem_cid=pubchem,
                inchikey=inchikey,
                drugbank_id=None,  # CoderData does not provide DrugBank IDs
            )
        )

    # ---- Tranche ----
    cohort_subtypes = tuple(sorted({s for s in patient_subtype.values() if s}))
    content_hash = _content_hash(
        name=name,
        coderdata_version=cd.__version__,
        cancer_type_filter=cancer_type_filter,
        matched_patients=matched_patients,
        drug_ids=sorted(experiments_cohort["improve_drug_id"].astype(str).unique()),
    )
    tranche = Tranche(
        tranche_id=tranche_id,
        source=f"coderdata_{name}",
        version=f"coderdata_{cd.__version__}",
        ingestion_date=ingestion_date or date.today(),
        patient_count=len(patients),
        sample_count=len(sample_objs),
        drug_count=len({a.drug_id for a in drug_assays}),
        subtypes=cohort_subtypes,
        content_hash=content_hash,
        description=(
            f"CoderData dataset '{name}' (package v{cd.__version__}), "
            "TPM transcriptomics + drug-response experiments, patient-level "
            "matched cohort. Drug response attaches to whichever sample CoderData "
            "carries it on (Tumor for sarcoma, cell line for gdscv2)."
            + (
                f" cancer_type filter: {cancer_type_filter!r}."
                if cancer_type_filter is not None
                else ""
            )
        ),
    )

    if skipped_metrics:
        adata.uns["skipped_response_metrics"] = skipped_metrics  # type: ignore[assignment]

    return CoderDataBundle(
        tranche=tranche,
        patients=patients,
        samples=sample_objs,
        expression=adata,
        drug_assays=drug_assays,
        baseline_expression=baseline_expression,
    )


def _content_hash(
    *,
    name: str,
    coderdata_version: str,
    cancer_type_filter: list[str] | None,
    matched_patients: list[str],
    drug_ids: list[str],
) -> str:
    """Deterministic sha256 of the tranche-defining inputs."""
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(coderdata_version.encode())
    if cancer_type_filter is not None:
        for ct in sorted(cancer_type_filter):
            h.update(ct.encode())
    for pat in matched_patients:
        h.update(pat.encode())
    for did in drug_ids:
        h.update(did.encode())
    return h.hexdigest()
