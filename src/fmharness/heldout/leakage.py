"""What each model version saw before rung 1 ran: the leakage records (design.md section 7).

Three things decide whether a rung 1 score is evidence about prediction rather than recall, and
each is written into a ``LeakageProfile`` record per model version:

* **The sci-Plex fine-tune saw one of the grid's lines with five of its drugs.** ``bc_large.ckpt``
  was fine-tuned on the sci-Plex screen, which includes A549 (DepMap ``ACH-000681``) treated with
  the five compounds ``grid.SCIPLEX_CIDS`` names. Where those drugs are in rung 1's grid, the
  (line, drug) pairs are removed **for every model** -- not only for the fine-tune -- so that
  every model is still scored on the same pairs (invariant 2), and the count is reported.
* **Stack's pretraining may include untreated cells of these lines.** That is input, not answer: a
  model that has seen a line's baseline expression has seen what rung 1 hands every model as the
  line's description. No Stack version saw Tahoe's *treated* cells, which is what rung 1 predicts.
* **Nothing else was fitted on the grid.** The remaining models (the drug average, chemistry only,
  ridge on expression, PCA, NMF, and the random stand-ins) are fitted inside the run, so their
  exposure is the run's own split and needs no record here.

``leakage_profiles`` refuses to write a record it cannot stand behind. If the sci-Plex line is in
the grid, the pairs found by PubChem id must be non-empty and must be exactly the pairs the grid
removed; a record that quietly reported no exposure would be indistinguishable from a model that
had none. When the line is not in the grid at all (a fixture, or a future grid without A549),
``sciplex_line_in_grid`` is False and the empty pair list means "not applicable here", which a
bare count of 0 could not say.
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# pandas call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fmharness.heldout.grid import SCIPLEX_CIDS, SCIPLEX_LINE, Grid, sciplex_exposed_pairs

#: Each Stack version and the checkpoint it is the mean embedding of (design.md section 4).
STACK_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("stack_base", "bc_large.ckpt"),
    ("stack_cytokine", "bc_large_aligned.ckpt"),
    ("stack_drug", "finetuned-epoch=5-val_loss=6.1078.ckpt"),
)

#: The one version fine-tuned on the sci-Plex screen, and so the one with drug exposure.
SCIPLEX_VERSION = "stack_drug"

#: What Stack's pretraining corpus can and cannot contain, in the record's own words.
PRETRAINING_NOTE = (
    "Stack's pretraining may include untreated cells of these lines: input, not answer. "
    "No Stack version saw Tahoe's treated cells, which is what rung 1 predicts."
)


class LeakageProfile(BaseModel):
    """One model version's exposure to rung 1's grid, before the run.

    ``exposed_pairs`` are the grid's (line, drug) pairs this version was fitted on before rung 1
    -- the sci-Plex fine-tune's A549 pairs -- and ``n_exposed_pairs`` their count, the number
    design section 7 asks to be reported. ``sciplex_line_in_grid`` says whether the exposed line
    is in this grid at all, so an empty pair list is never ambiguous.
    ``excluded_pairs_removed_for_every_model`` records the policy the run actually applied: the
    pairs come out for every model, not only for the one that saw them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_version: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    saw_tahoe_treated_cells: bool
    untreated_cells_may_be_in_pretraining: bool
    pretraining_note: str = Field(min_length=1)
    sciplex_fine_tuned: bool
    exposed_line: str | None
    exposed_cids: tuple[str, ...]
    sciplex_line_in_grid: bool
    exposed_pairs: tuple[tuple[str, str], ...]
    n_exposed_pairs: int = Field(ge=0)
    excluded_pairs_removed_for_every_model: bool


def leakage_profiles(grid: Grid, drug_metadata: pd.DataFrame) -> list[LeakageProfile]:
    """One ``LeakageProfile`` per Stack version, in ``STACK_CHECKPOINTS`` order.

    The sci-Plex pairs are found the same way the grid found them: grid drugs matched to
    ``drug_metadata`` on ``str.strip()``, kept when their ``pubchem_cid`` is one of
    ``SCIPLEX_CIDS``, paired with ``SCIPLEX_LINE``.

    Raises ``ValueError`` when the sci-Plex line is in the grid but no pair is found (the drug
    crosswalk did not match, so the run removed nothing it should have), and when the pairs found
    are not the pairs the grid excluded (the grid and this record disagree about what was
    removed). Both would otherwise be written as a clean "no exposure".
    """
    exposed = sciplex_exposed_pairs(grid, drug_metadata)
    line_in_grid = SCIPLEX_LINE in grid.lines
    if line_in_grid:
        if not exposed:
            raise ValueError(
                f"{SCIPLEX_LINE} is one of the grid's lines but no grid drug carries one of the "
                f"sci-Plex PubChem ids {SCIPLEX_CIDS}; the drug-metadata crosswalk matched "
                "nothing, so no leaked pair was removed"
            )
        if tuple(sorted(exposed)) != tuple(sorted(grid.excluded_pairs)):
            raise ValueError(
                f"the pairs the sci-Plex fine-tune saw {exposed} are not the pairs the grid "
                f"excluded {grid.excluded_pairs}; the run removed a different set from the one "
                "this record would claim"
            )

    profiles: list[LeakageProfile] = []
    for version, checkpoint in STACK_CHECKPOINTS:
        fine_tuned = version == SCIPLEX_VERSION
        pairs = exposed if fine_tuned else ()
        profiles.append(
            LeakageProfile(
                model_version=version,
                checkpoint=checkpoint,
                saw_tahoe_treated_cells=False,
                untreated_cells_may_be_in_pretraining=True,
                pretraining_note=PRETRAINING_NOTE,
                sciplex_fine_tuned=fine_tuned,
                exposed_line=SCIPLEX_LINE if fine_tuned else None,
                exposed_cids=SCIPLEX_CIDS if fine_tuned else (),
                sciplex_line_in_grid=line_in_grid,
                exposed_pairs=pairs,
                n_exposed_pairs=len(pairs),
                excluded_pairs_removed_for_every_model=True,
            )
        )
    return profiles


def leakage_records(profiles: list[LeakageProfile]) -> list[dict[str, Any]]:
    """The profiles as JSON-ready dictionaries, in the order given -- one place that knows how a
    record is serialized, so what is written is what ``LeakageProfile`` validates."""
    return [profile.model_dump(mode="json") for profile in profiles]


__all__ = [
    "PRETRAINING_NOTE",
    "SCIPLEX_VERSION",
    "STACK_CHECKPOINTS",
    "LeakageProfile",
    "leakage_profiles",
    "leakage_records",
]
