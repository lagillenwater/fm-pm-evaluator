"""The rung 1 grid: which (line, drug) pairs are scored, which are removed, and the ceiling.

Design.md section 2 fixes the grid at dose 5 uM to 107 drugs x 50 lines = 5,350 pairs: every
drug rung 0 measured, at that dose, in every one of the 50 lines. Section 7 removes two pairs
for leakage (the sci-Plex fine-tune saw ACH-000681 with five named drugs), and section 6 reads
the ceiling from rung 0's promoted per-pair table, limited to this grid. This module builds the
grid from rung 0's own tables (never a hand-typed list), matches Tahoe's screen drug names to
its drug-metadata table, finds the leaked pairs by PubChem CID, computes the ceiling, and writes
a restriction record that pins every input byte the grid was built from.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from fmharness.heldout.records import sha256_file
from fmharness.statistics import spearman_brown

#: Rung 1's fixed dose (design.md section 2).
DOSE_UM = 5.0

#: PubChem CIDs of the five Tahoe drugs the sci-Plex fine-tune saw in A549 (ACH-000681),
#: per design.md section 7.
SCIPLEX_CIDS: tuple[str, ...] = ("6918289", "11626560", "104741", "11707110", "3385")

#: The line the sci-Plex fine-tune saw.
SCIPLEX_LINE = "ACH-000681"


@dataclass(frozen=True)
class Grid:
    """The scored lines and drugs, the screen-to-metadata name crosswalk, and leaked pairs.

    ``lines`` and ``drugs`` are sorted tuples of the exact strings rung 0 used (including the
    literal ``"NA"`` line and any drug names carrying trailing whitespace). ``metadata_name``
    maps each grid drug's screen name to its name in the drug-metadata table (empty until
    ``attach_drug_metadata`` runs). ``excluded_pairs`` are (line, drug) pairs removed for
    leakage, empty until set.
    """

    lines: tuple[str, ...]
    drugs: tuple[str, ...]
    metadata_name: dict[str, str]
    excluded_pairs: tuple[tuple[str, str], ...]


def _at_dose(per_pair: pd.DataFrame, dose: float) -> pd.DataFrame:
    doses = cast(Any, pd.to_numeric(cast(Any, per_pair)["dose"], errors="coerce"))
    return cast(pd.DataFrame, per_pair[doses == dose])


def grid_from_rung0(per_pair: pd.DataFrame, dose: float = DOSE_UM) -> Grid:
    """The complete line x drug grid at ``dose``: lines are rung 0's distinct ``patient``
    values; drugs are those with an entry for every one of those lines.

    Raises ``ValueError`` if the resulting grid is not complete -- every line paired with
    every grid drug exactly once. Real, well-formed data cannot fail this (a drug's set of
    covered lines can only equal the full line set, never a same-size subset that omits one
    and duplicates another), so a raise here means a genuine data defect: a duplicate row,
    or a line-name inconsistency (e.g. stray whitespace) masking a gap.
    """
    at_dose = _at_dose(per_pair, dose)
    patients = cast(Any, at_dose)["patient"]
    drug_col = cast(Any, at_dose)["drug"]
    lines = tuple(sorted(str(v) for v in patients.unique().tolist()))
    per_drug_lines = cast(Any, at_dose.groupby("drug")["patient"].nunique())
    drug_names: list[str] = [str(v) for v in per_drug_lines.index.tolist()]
    drug_line_counts: list[int] = [int(v) for v in per_drug_lines.tolist()]
    drugs = tuple(
        sorted(
            name
            for name, count in zip(drug_names, drug_line_counts, strict=True)
            if count == len(lines)
        )
    )

    grid_rows = at_dose[drug_col.isin(drugs) & patients.isin(lines)]
    expected = len(lines) * len(drugs)
    if len(grid_rows) != expected:
        raise ValueError(
            f"grid is not complete: expected {expected} rows for {len(lines)} lines x "
            f"{len(drugs)} drugs, found {len(grid_rows)} (a duplicate or missing row)"
        )

    return Grid(lines=lines, drugs=drugs, metadata_name={}, excluded_pairs=())


def _crosswalk(drug_metadata: pd.DataFrame) -> dict[str, str]:
    """Drug-metadata names keyed by themselves stripped of whitespace."""
    names: list[str] = [str(v) for v in cast(Any, drug_metadata)["drug"].tolist()]
    return {name.strip(): name for name in names}


def attach_drug_metadata(grid: Grid, drug_metadata: pd.DataFrame) -> Grid:
    """Match every grid drug to the drug-metadata table on ``str.strip()``.

    Raises ``ValueError`` naming any grid drug with no match (never silently drops a drug).
    """
    metadata_name_by_stripped = _crosswalk(drug_metadata)
    unmatched = [drug for drug in grid.drugs if drug.strip() not in metadata_name_by_stripped]
    if unmatched:
        raise ValueError(f"no drug-metadata match for {len(unmatched)} grid drugs: {unmatched}")
    metadata_name = {drug: metadata_name_by_stripped[drug.strip()] for drug in grid.drugs}
    return replace(grid, metadata_name=metadata_name)


def _cid_by_stripped_name(drug_metadata: pd.DataFrame) -> dict[str, str]:
    names: list[str] = [str(v) for v in cast(Any, drug_metadata)["drug"].tolist()]
    cids: list[str] = [str(v) for v in cast(Any, drug_metadata)["pubchem_cid"].tolist()]
    return {name.strip(): cid for name, cid in zip(names, cids, strict=True)}


def sciplex_exposed_pairs(
    grid: Grid,
    drug_metadata: pd.DataFrame,
    line: str = SCIPLEX_LINE,
    cids: tuple[str, ...] = SCIPLEX_CIDS,
) -> tuple[tuple[str, str], ...]:
    """The grid's (line, drug) pairs the sci-Plex fine-tune saw, by PubChem CID.

    Matches grid drugs to ``drug_metadata`` on ``str.strip()`` (same crosswalk as
    ``attach_drug_metadata``) and keeps those whose ``pubchem_cid`` is one of ``cids``, paired
    with ``line``. Returns a sorted tuple; empty if ``line`` is not in the grid.
    """
    if line not in grid.lines:
        return ()
    cid_by_stripped = _cid_by_stripped_name(drug_metadata)
    cid_set = set(cids)
    exposed = [(line, drug) for drug in grid.drugs if cid_by_stripped.get(drug.strip()) in cid_set]
    return tuple(sorted(exposed))


def ceiling_table(per_pair: pd.DataFrame, grid: Grid, dose_strata: pd.DataFrame) -> pd.DataFrame:
    """The grid's split-half ceiling for both gene sets, beside rung 0's promoted 5 uM row.

    For each gene set, averages rung 0's per-pair split-half r over the grid's pairs it
    scored on that gene set (non-missing ``r`` for all genes, ``r_responder`` for responding
    genes), lifts the mean with Spearman-Brown, and reports its square root -- design.md
    section 6's ceiling. The promoted columns are the matching row of rung 0's promoted
    ``per_triple`` dose-strata table at dose 5 uM, unchanged (not recomputed).
    """
    at_dose = _at_dose(per_pair, DOSE_UM)
    patients = cast(Any, at_dose)["patient"]
    drug_col = cast(Any, at_dose)["drug"]
    grid_rows = cast(Any, at_dose[patients.isin(grid.lines) & drug_col.isin(grid.drugs)])

    promoted = cast(Any, dose_strata)
    promoted_mask = (promoted["dose"] == "5.0") & (promoted["weighting"] == "per_triple")
    promoted_rows = promoted[promoted_mask].set_index("gene_set")

    rows: list[dict[str, float | str | int]] = []
    for gene_set, column, strata_gene_set in (
        ("responding", "r_responder", "responder"),
        ("all", "r", "all"),
    ):
        scored = grid_rows[column].dropna()
        split_half_r = float(scored.mean())
        sb = spearman_brown(split_half_r)
        promoted_r = float(promoted_rows.loc[strata_gene_set, "splithalf_mean_r"])
        promoted_sb = float(promoted_rows.loc[strata_gene_set, "spearman_brown_full"])
        rows.append(
            {
                "gene_set": gene_set,
                "pairs_scored_rung0": int(scored.shape[0]),
                "split_half_r": split_half_r,
                "sb": sb,
                "sqrt_sb": math.sqrt(sb),
                "promoted_r": promoted_r,
                "promoted_sb": promoted_sb,
                "promoted_sqrt_sb": math.sqrt(promoted_sb),
            }
        )
    return pd.DataFrame(rows)


def _sha256_lines(items: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def restriction_record(grid: Grid, sources: dict[str, Path]) -> dict[str, object]:
    """The record pinning exactly which pairs are scored, which are removed, and from what.

    ``sources`` maps a name to the path each input was read from; ``source_sha256`` records
    that file's sha256 so the grid's provenance can be checked byte-for-byte later.
    """
    return {
        "dose": DOSE_UM,
        "lines": list(grid.lines),
        "drugs": list(grid.drugs),
        "excluded_pairs": [list(pair) for pair in grid.excluded_pairs],
        "sha256_lines": _sha256_lines(grid.lines),
        "sha256_drugs": _sha256_lines(grid.drugs),
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
    }


def load_drug_metadata(path: Path) -> pd.DataFrame:
    """The Tahoe drug table from either the Hugging Face parquet or its offline CSV fixture.

    Normalizes ``pubchem_cid`` to an integer string (empty when missing) either way: the
    parquet stores it as a float (e.g. ``387447.0``), the CSV as a string that ``pandas``
    would otherwise infer as float too. Keeps only ``drug``, ``pubchem_cid``,
    ``canonical_smiles`` -- the columns the grid needs.
    """
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    raw_any = cast(Any, raw)
    numeric_cid = cast(Any, pd.to_numeric(raw_any["pubchem_cid"], errors="coerce"))
    cid = [("" if pd.isna(v) else str(int(v))) for v in numeric_cid.tolist()]
    return pd.DataFrame(
        {
            "drug": raw_any["drug"].tolist(),
            "pubchem_cid": cid,
            "canonical_smiles": raw_any["canonical_smiles"].tolist(),
        }
    )


__all__ = [
    "DOSE_UM",
    "SCIPLEX_CIDS",
    "SCIPLEX_LINE",
    "Grid",
    "attach_drug_metadata",
    "ceiling_table",
    "grid_from_rung0",
    "load_drug_metadata",
    "restriction_record",
    "sciplex_exposed_pairs",
]
