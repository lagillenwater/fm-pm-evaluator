"""Map Stack's gene vocabulary onto Soragni's measured genes (writes the panel).

data/static/stack_hvg_genes.txt is Stack's 15,012-symbol vocabulary, taken from
arcinstitute/Stack-Large on HuggingFace (basecount_1000per_15000max.pkl: the
union of the top-1000 HVGs per scBaseCount dataset, capped near 15,000). It is
tracked under data/static/ so the gene filter is pinned and ships with the repo.
To refresh it (only if Arc updates the model), download the file
`basecount_1000per_15000max.pkl` from the Stack-Large HuggingFace repo and write
its unpickled contents one symbol per line to data/static/stack_hvg_genes.txt.

Writes data/static/stack_soragni_gene_map.csv (stack_symbol, entrez_id, match):
the shared panel that puts expression, PCA, NMF, and Stack on the same genes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fmharness.data.loaders import load_coderdata_tranche
from fmharness.evaluation import build_sample_design
from fmharness.stack_panel import build_panel


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    stack = [
        g.strip()
        for g in (repo / "data/static/stack_hvg_genes.txt").read_text().splitlines()
        if g.strip()
    ]
    genes = pd.read_csv(repo / "data/raw/coderdata/genes.csv.gz")
    bundle = load_coderdata_tranche("sarcoma", repo)
    x_df, _ = build_sample_design(bundle, "organoid", "auc")
    measured = {int(c) for c in x_df.columns}

    panel = build_panel(genes, measured, stack)
    dest = repo / "data/static/stack_soragni_gene_map.csv"
    panel.to_csv(dest, index=False)

    n = len(stack)
    n_sym = int((panel["match"] == "symbol").sum())
    n_ali = int((panel["match"] == "alias").sum())
    print(f"Stack vocabulary:    {n}")
    print(f"  direct symbol:     {n_sym}")
    print(f"  alias-recovered:   {n_ali}")
    print(f"  final panel:       {len(panel)} / {n} = {len(panel) / n:.1%}")
    print(f"wrote {dest.relative_to(repo)}")


if __name__ == "__main__":
    main()
