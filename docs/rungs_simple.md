# Rungs 0-3, the simple run (2026-09-12/13)

A plain record of what was measured this weekend, outside the promotion machinery: one script and one
job per question, tables pulled from Alpine scratch, nothing promoted. Branches: `rung1-simple-tables`
(rung 0 tables, rung 1), `rung2-bridge` (real cells, rung 2), `rung3-platform` (rung 3). The two lines
with empty DE tables (ACH-000628, ACH-000311) are dropped from rung 3 onward; the line whose DepMap id is
the string `NA` from rung 1 onward.

## Rung 0 -- noise per drug and per line
Two group-bys of the promoted per-pair split-half table at 5 uM (`rung0_noise_per_drug.csv`,
`rung0_noise_per_line.csv`). On a condition's own DE genes the replicate correlation is ~0.6 (0.88 for the
strongest drugs); on any wide gene set ~0.1.

## Rung 1 -- can a new line's response be predicted from its baseline?
Held-out line, own-DE panel (within Stack's 14,346 genes), 5,055 conditions, 108 drugs x 49 lines.
Attainable bound 0.72. Drug mean over other lines r = 0.653 (0.90 of attainable); pca_adj 0.654, nmf_adj
0.648, knn 0.606. Stack in-context generation on REAL Tahoe cells (prompt/context/query all real):
0.158 / 0.105 direct, 0.356 / 0.351 as a departure from the mean. Nothing beats the drug mean; Stack's
generated response is essentially unrelated to the measured one (`results_v4_real`).

## Rung 2 -- do synthesised cells land where real cells land?
Real Tahoe cells (681,860 at 5 uM, `tahoe_real_cells/cache/real_cells_5um.h5ad`). Baseline: a line's
synthetic population is nearest its own real population for 39/49 lines but sits 5.4 replicate-distances
away. Treated states (5,046 conditions): own line 97%, own drug 16% (48% for the largest real shifts),
closer to real treated than to real control 71%, ratio 6.5. Identity survives the bridge; position does
not. The rung 1 Stack run on real cells (above) made the bridge unnecessary for this evaluation.

## Rung 3 -- what does the platform cost?
LINCS 2020 Level 3 (66 of the 108 drugs by name, 160 LINCS lines, 24 h, ~10 uM, plate-vehicle log2FC over
the 978 landmarks) -> a new Tahoe line, on the own-DE genes within the landmarks (median 183). Every rung 1
predictor fit on LINCS lands at r ~ 0.09 where the same predictor fit on Tahoe reaches 0.73 (retained 0.13;
0.20 above each platform's wrong-drug floor). The Tahoe-internal mean carries a screen-generic response:
a wrong drug's Tahoe mean scores 0.29. Stack on the pure LINCS set (vehicle pseudo-cells as prompt, treated
pseudo-cells as context, real Tahoe query) scores 0.12, the same as with real Tahoe context -- its output
is context-independent at the floor. Expected transfer measured directly: for the 8 lines on both platforms,
baselines match after within-platform standardisation (r 0.5-0.6, rank 1 of 160 in all 8) but the same
line's response to the same drug crosses at r = 0.09 (132 pairs), no better than the LINCS mean over other
lines. The platform pair itself sets the ceiling at ~0.09; rung 3's predictor sits on it. Drugs with large
canonical effects transfer (Trametinib 0.53, Artesunate 0.44, Neratinib 0.37); the median drug does not.
Dataset census: nothing public has more line x drug overlap with Tahoe than LINCS (MIX-seq: 18 lines, 2
drugs; PANACEA DREAM release: 11 lines, 32 concealed compounds).

Scripts: `scripts/rung1_simple_tables.py`, `scripts/tahoe_real_cells.py`, `scripts/rung2_bridge.py`,
`scripts/rung3_platform.py`, `scripts/rung3_baseline_similarity.py`; jobs under `scripts/alpine/`.
`docs/STATE.md` and `README.md` are not updated by this note.

## Rung 4 -- does the representation predict how much a drug kills a line?
PRISM Repurposing 19Q4 primary screen (public; viability log fold change at 2.5 uM, 5 days): 37 Tahoe
lines x 69 of the 108 drugs, 2,553 pairs, median 3 replicates. Target = the line-specific part (V minus the
drug mean over training lines); ceiling = replicate split-half of that quantity, per drug across lines:
mean 0.235, median 0.18, above 0.4 for 21 drugs, below 0 for 20 -- the line-specific killing is itself
only weakly reproducible in a single-dose screen. Attainable bound (sqrt of Spearman-Brown) 0.49.
Per-drug r across held-out lines, mean over 69 drugs (line-shuffled null in brackets): pca_adj 0.058 (-0.04),
stack_base 0.036 (0.06), knn 0.029 (0.01), nmf_adj 0.024 (0.05), tahoe_prolif 0.017 (-0.02), tahoe_n_de
-0.006, stack_gen_prolif cytokine 0.026 / sciplex 0.014 (0.01 / 0.02). SE across drugs ~0.02. On the 21
drugs with reproducible line-specific killing: stack_base 0.14, knn 0.10, pca_adj 0.10, the response
summaries 0.04-0.05. The drug mean alone correlates 0.83 with viability globally. No predictor -- baseline
representation, measured transcriptional response, or Stack's generated response -- recovers a usable part
of the line-specific killing; the best is ~0.1 of the attainable bound. (`scripts/rung4_viability.py`;
results on scratch `rung4_viability/results`, worktree `results/rung4-viability/`.)
