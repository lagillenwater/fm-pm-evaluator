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
mean 0.235 (0.26 with each line's mean removed), above 0.4 for 21 drugs -- line-specific killing is itself
only weakly reproducible in a single-dose screen; attainable bound ~0.49. The drug mean alone correlates
0.83 with viability globally.
Leave-one-line-out predictions from a many-feature regression are anti-correlated with the held-out value
by construction (-0.15 on a null synthetic), so every entry is read against a MATCHED null: viability
shuffled across lines within each drug and the whole procedure rerun 10 times (`--permute-seed`; p = fraction
of permutations at or above, floor 1/11 = 0.09).
Layout: rows = what predicted the held-out line's expression response in rung 1 (mean, kNN, PCA, NMF,
Stack-base, Stack-cytokine, Stack-sci-plex) plus the measured response; one table per readout that turns a
response into a viability prediction: proliferation (minus the mean z over Hallmark E2F + G2-M, no fitting),
ridge and lasso (per drug over the training lines' DE genes, genes standardised on the training lines, one
fit at a fixed penalty -- ridge 1, lasso a tenth of alpha_max -- no tuning). Every row's additive base is the
block-wide drug mean, so the mean row is a per-drug constant: its proliferation residual is 0 and ridge /
lasso have nothing to fit (blank). Columns: overall r, its p, interaction r (each line's mean removed), its
p, and each minus the null mean.
Stack-base row: the base encoder's embedding of the line's real DMSO cells (all 1,600 dimensions) mapped into
gene by line space by a ridge from the centred embedding to the training lines' response departures (one fixed
penalty, the mean squared embedding norm; the base checkpoint has no decoder, so this map is ours, leave-line-out);
the predicted departure is a similarity-weighted combination of the other lines' measured departures, gene for gene.
Results (observed minus null / p), 37 lines x 69 drugs, ceiling 0.235 overall / 0.262 interaction:
  proliferation  measured +0.028 / 0.09 (interaction +0.030 / 0.09); stack_cytokine +0.022 / 0.09 (+0.015 /
                 0.45); pca +0.018 / 0.27; knn ~0; stack_base -0.021 / 0.82 (+0.007 / 0.27); stack_sciplex ~0;
                 nmf -0.04.
  ridge          measured +0.066 / 0.09 (+0.083 / 0.09); stack_cytokine +0.069 / 0.09 (+0.087 / 0.09);
                 stack_sciplex +0.057 / 0.18 (+0.072 / 0.09); pca +0.024 / 0.36, nmf +0.024 / 0.18 (+0.036 /
                 0.09); knn -0.01; stack_base -0.031 / 0.91 (-0.026 / 0.82).
  lasso          measured +0.085 / 0.09 (+0.088 / 0.09); stack_base +0.021 / 0.09 (+0.040 / 0.09);
                 stack_cytokine +0.029 / 0.18; pca +0.025 / 0.36; nmf +0.022 / 0.27 (+0.028 / 0.09);
                 stack_sciplex ~0; knn -0.02.
Null sign depends on the row. Under the ridge readout the null mean is -0.02 to -0.03 for the measured and
generated rows (the leave-one-out anti-correlation) but +0.02 for knn and +0.066 (SD 0.03 over 10 perms) for
stack_base. Mechanism: the readout's targets are centred on the training lines, so each carries +V_i/(n-1) of
the held-out line's own viability; the sign of the resulting bias follows the sign of the summed similarity
between the held-out line's features and the training lines' features, negative for rows whose features are
their own (measured, generated), positive for rows whose features are similarity-weighted combinations of the
other lines' measured departures (knn, stack_base). The matched null removes it either way; it is why the
observed values are not read on their own.
Reading: the measured response carries a small line-specific signal into viability (0.03 through the
proliferation programme, 0.07-0.09 through a regression on its DE genes; 30-35% of the split-half ceiling).
Of the predicted responses, only Stack's generated ones reach it under ridge (cytokine +0.07, sci-plex
+0.06; at the p floor), and none under the proliferation readout; PCA, NMF and kNN departures add 0.02 or
less; the Stack-base row is inconsistent across readouts (-0.03 ridge, +0.02 lasso, ~0 proliferation) and
carries nothing reliable. Ten permutations floor p at 0.09. (`scripts/rung4_viability.py`,
`scripts/rung4_combine.py`; results on scratch `rung4_viability/results` + `perm_1..10`, worktree
`results/rung4-viability/v5/`.)
