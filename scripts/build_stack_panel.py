"""Map Stack's gene vocabulary onto Soragni's measured genes.

Stack names genes by HGNC symbol; CoderData indexes expression by Entrez id.
Most Stack symbols match a measured gene directly through CoderData's canonical
Entrez->symbol table. The rest are largely older or aliased symbols (e.g. AARS,
now AARS1), which CoderData records in its alias column (other_id). We recover
those by matching the Stack symbol against the alias of a measured Entrez id,
keeping only unambiguous, not-already-used matches.

Writes data/reference/stack_soragni_gene_map.csv (stack_symbol, entrez_id,
match), the shared gene panel used to put expression, PCA, NMF, and Stack on the
same genes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fmharness.data.loaders import load_coderdata_tranche
from fmharness.evaluation import build_sample_design


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    stack = [
        g.strip()
        for g in (repo / "data/reference/stack_hvg_genes.txt").read_text().splitlines()
        if g.strip()
    ]
    genes = pd.read_csv(repo / "data/raw/coderdata/genes.csv.gz")
    bundle = load_coderdata_tranche("sarcoma", repo)
    x_df, _ = build_sample_design(bundle, "organoid", "auc")
    measured = {int(c) for c in x_df.columns}

    # canonical Entrez -> symbol, then symbol -> Entrez among measured genes
    canon = (
        genes.dropna(subset=["gene_symbol"])
        .drop_duplicates("entrez_id")
        .set_index("entrez_id")["gene_symbol"]
        .astype(str)
    )
    sym2ent: dict[str, int] = {}
    for ent, sym in canon[canon.index.isin(measured)].items():
        sym2ent.setdefault(sym, int(ent))

    direct = {s: sym2ent[s] for s in stack if s in sym2ent}
    missing = [s for s in stack if s not in sym2ent]

    # alias (other_id) -> measured Entrez ids
    al = genes.dropna(subset=["other_id"])
    al = al[al["entrez_id"].isin(measured)]
    alias2ent: dict[str, set[int]] = {}
    for oid, ent in zip(al["other_id"].astype(str), al["entrez_id"].astype(int), strict=True):
        alias2ent.setdefault(oid, set()).add(ent)

    used = set(direct.values())
    recovered: dict[str, int] = {}
    ambiguous: list[str] = []
    for s in missing:
        cand = alias2ent.get(s, set()) - used  # not already in the panel
        if len(cand) == 1:
            e = next(iter(cand))
            recovered[s] = e
            used.add(e)
        elif len(cand) > 1:
            ambiguous.append(s)

    rows = [{"stack_symbol": s, "entrez_id": e, "match": "symbol"} for s, e in direct.items()]
    rows += [{"stack_symbol": s, "entrez_id": e, "match": "alias"} for s, e in recovered.items()]
    out = pd.DataFrame(rows).sort_values("stack_symbol")
    dest = repo / "data/reference/stack_soragni_gene_map.csv"
    out.to_csv(dest, index=False)

    n = len(stack)
    print(f"Stack vocabulary:      {n}")
    print(f"  direct symbol match: {len(direct)}")
    print(f"  alias-recovered:     {len(recovered)}")
    print(f"  ambiguous (skipped): {len(ambiguous)} (e.g. {ambiguous[:6]})")
    print(f"  still missing:       {n - len(direct) - len(recovered) - len(ambiguous)}")
    print(f"final panel:           {len(out)} / {n} = {len(out) / n:.1%}")
    print(f"wrote {dest.relative_to(repo)}")


if __name__ == "__main__":
    main()
