"""Map Stack's gene vocabulary onto a CoderData Entrez-id gene set.

Pure mapping logic shared by ``scripts/build_stack_panel.py`` and the
reproducibility check. A Stack symbol maps to a measured gene either by a direct
match on CoderData's canonical ``gene_symbol``, or, failing that, by an
unambiguous alias in CoderData's ``other_id`` column. An Entrez id is never used
twice, so a symbol whose gene is already mapped under its canonical name is not
re-added under an alias.
"""

from __future__ import annotations

import pandas as pd


def build_panel(genes: pd.DataFrame, measured: set[int], stack_genes: list[str]) -> pd.DataFrame:
    """Return the Stack<->Soragni panel (columns: stack_symbol, entrez_id, match).

    ``genes`` is CoderData's gene table (entrez_id, other_id, gene_symbol,
    other_id_source). ``measured`` is the set of Entrez ids present in the
    expression matrix. ``stack_genes`` is Stack's vocabulary (HGNC symbols).
    The result is sorted by symbol so it is byte-stable across runs.
    """
    canon = genes.dropna(subset=["gene_symbol"]).drop_duplicates("entrez_id")
    canon = canon[canon["entrez_id"].isin(list(measured))]
    sym2ent: dict[str, int] = {}
    for sym, ent in zip(canon["gene_symbol"].astype(str), canon["entrez_id"], strict=True):
        sym2ent.setdefault(str(sym), int(ent))

    direct = {s: sym2ent[s] for s in stack_genes if s in sym2ent}
    missing = [s for s in stack_genes if s not in sym2ent]

    al = genes.dropna(subset=["other_id"])
    al = al[al["entrez_id"].isin(list(measured))]
    alias2ent: dict[str, set[int]] = {}
    for oid, ent in zip(al["other_id"].astype(str), al["entrez_id"], strict=True):
        alias2ent.setdefault(str(oid), set()).add(int(ent))

    used = set(direct.values())
    recovered: dict[str, int] = {}
    for s in missing:
        cand = alias2ent.get(s, set()) - used  # not already in the panel
        if len(cand) == 1:
            e = next(iter(cand))
            recovered[s] = e
            used.add(e)

    rows = [{"stack_symbol": s, "entrez_id": e, "match": "symbol"} for s, e in direct.items()]
    rows += [{"stack_symbol": s, "entrez_id": e, "match": "alias"} for s, e in recovered.items()]
    return pd.DataFrame(rows).sort_values("stack_symbol").reset_index(drop=True)
