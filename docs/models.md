# Models under evaluation

Operational description of every model/representation the harness scores, kept next to
the code so it stays in sync. For the encoder interface see [adapter_contract.md](adapter_contract.md);
for how the comparison is read see the Results section of the [README](../README.md). The
scholarly description of each model (architecture, citations, rationale) lives in the
companion manuscript (greenelab/fm-pm-eval-manuscript), which cross-references this file.

## Two layers: representation and head

A prediction is `response(sample, drug) = a_drug + f(representation(sample))`. Two choices
are independent:

- **Representation** — the per-sample vector a model produces: PCA/NMF of expression, or a
  foundation-model embedding (Stack). Swapping the representation is how we ask "what does the
  encoder capture."
- **Head** — the function `f` fit on top: a linear ridge slope (`SimpleProbe`) or an RBF
  kernel ridge (`KernelProbe`). Swapping the head is how we ask whether a finding is
  *head-invariant* (a linear head could under-read a representation that only pays off
  nonlinearly).

Both heads share the per-drug mean and the PCA/NMF reduction
([`probe/base.py`](../src/fmharness/probe/base.py)) and the same `fit(emb, drugs, y)` /
`predict_parts(emb, drugs) -> (base, residual)` contract, so any representation can be scored
under either head apples-to-apples. The bilinear and biomarker models below are standalone
(they do not use this head interface).

## Catalogue

| Model | Representation | Head / form | Input + normalization | Where |
|---|---|---|---|---|
| Expression PCA | top-k PCA of log1p CPM expression | linear ridge (`SimpleProbe`) | bulk RNA-seq, CPM then log1p; per-drug ridge over k PCs | [`probe/simple.py`](../src/fmharness/probe/simple.py) |
| Expression NMF | non-negative gene programs of log1p CPM | linear ridge | same; NMF requires non-negative input | `probe/simple.py` (`reducer="nmf"`) |
| Stack-Large | 1600-dim Stack embedding (PCA-reduced) | linear ridge | CPM expression fed to Stack as pseudo-cells; embedding CSV indexed by sample id | [`stack_panel.py`](../src/fmharness/stack_panel.py), `probe/simple.py` |
| Kernel (any representation) | same PCA/NMF reduction, score-standardized | RBF kernel ridge (`KernelProbe`) | identical to the linear head's input; `(alpha, gamma)` by per-drug leave-one-out | [`probe/kernel.py`](../src/fmharness/probe/kernel.py) |
| Bilinear (PharmaFormer-lite) | expression PCA `z` + Morgan-fingerprint PCA `g` | ridge on `[z, g, z⊗g]` | log1p CPM expression + 1024-bit Morgan fingerprint; predicts unseen drugs via chemistry | [`scripts/transfer_pharmaformer_lite.py`](../scripts/transfer_pharmaformer_lite.py), [`bilinear.py`](../src/fmharness/bilinear.py) |
| Biomarker | pre-specified genomic rules (e.g. PIK3CA→everolimus) | rule offsets on the GDSC2 drug-mean prior | WES SNV/CNV calls + drug identity | [`scripts/biomarker_anchored.py`](../scripts/biomarker_anchored.py) |
| Drug-mean baseline | none | `a_drug` only (`n_components=0`) | response labels only | `probe/simple.py` |

## Shared guarantees

- **Graceful degradation.** Every learned head shrinks its embedding term toward 0 when the
  representation is uninformative (ridge penalty / large kernel alpha), so an uninformative
  model reduces to the drug-mean baseline rather than injecting noise — this is what makes a
  null result a real null, not a weak model.
- **`base + residual` split.** `predict_parts` returns the per-drug mean and the embedding
  residual separately so the metrics (`interaction_rho`, `within_drug_rho` in
  [`evaluation.py`](../src/fmharness/evaluation.py)) score the embedding part alone, avoiding
  the leave-one-out drug-base artifact.
- **Coverage.** The per-drug transfers (expression, Stack) score only drugs shared between
  GDSC2 and Soragni (~15–21 by PubChem CID); the bilinear scores the full fingerprinted panel.
  Uncovered drugs fall back to the GDSC2 drug-mean.
- **Determinism.** All heads are closed-form (RidgeCV / KernelRidge) and seeded; repeated fits
  are bit-identical.
