"""Bridge the native Soragni / GDSC2 loaders onto the downstream bundle contract.

The native loaders (``load_soragni`` / ``load_gdsc2_sarcoma``) parse raw artifacts
into validated schema objects, but they key expression on the namespace each
source ships (Ensembl gene_id for Soragni, HGNC symbol for GDSC2) and they don't
carry the ``improve_sample_id`` / ``model_type`` sample metadata that
``build_sample_design`` reads. Everything downstream -- the Stack gene panel, the
PCA/NMF baselines, the cross-substrate transfer -- assumes a single ``CoderDataBundle``
shape: expression with **Entrez** ``var_names``, ``obs_names`` equal to each
sample's ``improve_sample_id``, and ``model_type`` on the sample metadata.

This module adapts the native bundles to that contract so the analysis scripts
need only swap ``load_coderdata_tranche`` for ``load_tranche``. ``load_tranche``
dispatches sarcoma/gdscv2 to the native loaders and falls back to CoderData for
any other dataset (liver, bladder, ...), which we have not re-implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.data.loaders.coderdata import (
    CODERDATA_LOCAL_PATH_DEFAULT,
    CoderDataBundle,
    load_coderdata_tranche,
)
from fmharness.data.loaders.gdsc2_sarcoma import GDSC2SarcomaBundle, load_gdsc2_sarcoma
from fmharness.data.loaders.soragni import SoragniBundle, load_soragni
from fmharness.schema import Sample

# Soragni specimen marker -> the model_type vocabulary build_sample_design filters on.
_SORAGNI_MODEL_TYPE = {"Organoid": "patient derived organoid", "Tumor": "tumor"}


def _collapse_to_entrez(
    values: Any, obs_names: list[str], entrez: Sequence[int | None]
) -> pd.DataFrame:
    """Relabel gene columns to Entrez and sum any that collapse to one Entrez id.

    Genes without a positive Entrez id are dropped. Multiple source genes mapping
    to the same Entrez (e.g. two Ensembl ids) are summed -- the gene's total
    expression -- which is the right reduction for count-derived values (CPM,
    median-of-ratios) where additivity holds within a sample.
    """
    cols = np.asarray([e if (e is not None and int(e) > 0) else -1 for e in entrez])
    keep = cols > 0
    x = np.asarray(values, dtype=np.float64)[:, keep]
    df = pd.DataFrame(
        x, index=pd.Index([str(o) for o in obs_names]), columns=pd.Index(cols[keep].astype(str))
    )
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).sum().T
    return df


def _rebuild_samples(samples: list[Sample], model_type_of: dict[str, str] | str) -> list[Sample]:
    """Copy samples, stamping ``improve_sample_id`` (= sample_id) and ``model_type``."""
    out: list[Sample] = []
    for s in samples:
        mt = model_type_of if isinstance(model_type_of, str) else model_type_of[s.sample_id]
        out.append(
            Sample(
                sample_id=s.sample_id,
                patient_id=s.patient_id,
                tranche_id=s.tranche_id,
                passage=s.passage,
                replicate=s.replicate,
                metadata={**s.metadata, "improve_sample_id": s.sample_id, "model_type": mt},
            )
        )
    return out


def _build_expression(
    expr_df: pd.DataFrame, src: ad.AnnData, raw_df: pd.DataFrame | None = None
) -> ad.AnnData:
    """Assemble the adapted AnnData: Entrez var, obs_names = improve_sample_id."""
    var = pd.DataFrame(index=expr_df.columns, data={"entrez_id": [int(c) for c in expr_df.columns]})
    var.index.name = "entrez_id_str"
    obs = cast(pd.DataFrame, src.obs).loc[expr_df.index].copy()
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=expr_df.to_numpy(dtype=np.float64), obs=obs, var=var)
    adata.uns.update(dict(src.uns.items()))
    if raw_df is not None:
        adata.layers["raw_counts"] = raw_df.loc[expr_df.index, expr_df.columns].to_numpy()
    return adata


def adapt_gdsc2(bundle: GDSC2SarcomaBundle) -> CoderDataBundle:
    """GDSC2 native bundle -> CoderDataBundle (Entrez var, raw_counts layer kept)."""
    entrez = [int(e) for e in bundle.expression.var["entrez"]]
    obs_names = [str(o) for o in bundle.expression.obs_names]
    expr_df = _collapse_to_entrez(bundle.expression.X, obs_names, entrez)
    raw_df = _collapse_to_entrez(bundle.expression.layers["raw_counts"], obs_names, entrez)
    expr = _build_expression(expr_df, bundle.expression, raw_df=raw_df)
    samples = _rebuild_samples(bundle.samples, "cell line")
    return CoderDataBundle(
        tranche=bundle.tranche,
        patients=bundle.patients,
        samples=samples,
        expression=expr,
        drug_assays=bundle.drug_assays,
        baseline_expression=bundle.baseline_expression,
    )


def adapt_soragni(bundle: SoragniBundle, repo_root: Path) -> CoderDataBundle:
    """Soragni native bundle -> CoderDataBundle (Ensembl gene_id mapped to Entrez)."""
    genes = pd.read_csv(repo_root / "data/raw/coderdata/genes.csv.gz")
    ens = cast(pd.DataFrame, genes[genes["other_id_source"] == "ensembl_gene"]).dropna(
        subset=["other_id"]
    )
    ens2ent = dict(zip(ens["other_id"].astype(str), ens["entrez_id"].astype(int), strict=True))
    entrez = [ens2ent.get(str(g)) for g in bundle.expression.var_names]
    obs_names = [str(o) for o in bundle.expression.obs_names]
    expr_df = _collapse_to_entrez(bundle.expression.X, obs_names, entrez)
    expr = _build_expression(expr_df, bundle.expression)
    model_type_of = {
        s.sample_id: _SORAGNI_MODEL_TYPE[str(s.metadata["specimen"])] for s in bundle.samples
    }
    samples = _rebuild_samples(bundle.samples, model_type_of)
    return CoderDataBundle(
        tranche=bundle.tranche,
        patients=bundle.patients,
        samples=samples,
        expression=expr,
        drug_assays=bundle.drug_assays,
        baseline_expression=bundle.baseline_expression,
    )


# Dataset-name aliases accepted for the two re-implemented cohorts.
_SORAGNI_NAMES = {"sarcoma", "soragni"}
_GDSC2_NAMES = {"gdscv2", "gdsc2_sarcoma"}


def load_tranche(
    name: str,
    repo_root: Path,
    *,
    cancer_type_filter: list[str] | None = None,
    ingestion_date: date | None = None,
    local_path: str = CODERDATA_LOCAL_PATH_DEFAULT,
) -> CoderDataBundle:
    """Load a tranche, preferring our own raw-artifact loaders for the MVP cohorts.

    ``sarcoma`` -> Soragni PDTO (CPM, length-free); ``gdscv2`` -> GDSC2 (DepMap
    raw counts -> DESeq2 median-of-ratios, with raw counts retained), the **full
    cell-line panel** by default. Both are returned in the ``CoderDataBundle``
    shape ``build_sample_design`` expects. Passing a non-None ``cancer_type_filter``
    is the signal to restrict GDSC2 to sarcoma lineages (the native loader supports
    a sarcoma restriction, not arbitrary cancer-type slicing). Any other ``name``
    falls through to CoderData unchanged.
    """
    if name in _SORAGNI_NAMES:
        return adapt_soragni(load_soragni(repo_root, ingestion_date=ingestion_date), repo_root)
    if name in _GDSC2_NAMES:
        native = load_gdsc2_sarcoma(
            repo_root, sarcoma_only=cancer_type_filter is not None, ingestion_date=ingestion_date
        )
        return adapt_gdsc2(native)
    return load_coderdata_tranche(
        name,
        repo_root,
        cancer_type_filter=cancer_type_filter,
        ingestion_date=ingestion_date,
        local_path=local_path,
    )
