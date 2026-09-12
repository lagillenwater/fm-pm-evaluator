# Rung 1 — drift audit

**Date** 2026-09-11.
**Commit audited** `55adfb7` (`rung1-held-out-prediction-work`, cut from `dff524f`).
**Auditor** A fresh reader (`docs/audit.md`, "Who runs it"): did not write this code, this design or
this plan, and worked from the documents and the tree rather than from the conversation that
produced them.
**Procedure** [`docs/audit.md`](../../audit.md), followed as written.

> **The run has not happened. 4 of the 197 claims could not be checked at all, and 55 more are
> checked on their code with the artifact still to come.**
> Alpine's `/scratch/alpine` is unmounted after a data-centre power event, so no rung 1 job has
> produced an artifact: there are no score tables, no comparison tables, no control tables, no
> figures, no parameter sidecar, no logs. Claims about the code, the documents, the jobs and the
> tests are auditable now and are verdicted normally. Claims whose entire content is a run output
> are verdicted **PENDING-RUN** (4 of them: D107, P36, P47, P48) and listed with the evidence each
> will need. A further **55 claims carry a verdict now on the code and documents that will produce
> the artifact**; their realized form arrives with the run, and every one is named in the *Confirms*
> column of the [PENDING-RUN artifact checklist](#pending-run-the-artifact-checklist), so the audit
> delta after the run has a checklist rather than a re-enumeration. No claim was verdicted ALIGNED
> on the strength of a fixture — a fixture is not the artifact a claim about the run is about.

## Method

**Read in full.** `docs/audit.md`; `.superpowers/sdd/plan/task-13-brief.md`; the four task
documents (`design.md`, `plan.md`, `decisions.md`, `verification.md`); `docs/SPEC.md`; the branch
diff of `README.md`, `docs/STATE.md`, `docs/DATA.md`, `pyproject.toml`; the ten
`src/fmharness/heldout/` modules plus `statistics.py` and `tahoe.py`; `scripts/verify_rung1.py`
(the parts keyed to the grid and to the design's constants); the eleven rung 1 job scripts and
`submit_rung1_chain.sh`; and the two notebooks as JSON.

**Recomputed rather than read.** Every mechanical claim went to a command; the command and its
output are the evidence in the tables below.

```
$ git diff dff524f...HEAD --stat | tail -1
 69 files changed, 22355 insertions(+), 88 deletions(-)
```

The grid, the dose rationale and both ceiling rows, re-derived from rung 0's **committed promoted
tables** (these are rung 0's artifacts, present in the tree, so §2 and §6 are checkable now even
though rung 1 has not run):

```
$ uv run python -  # read results/rung0-assay-reliability/rung0_per_pair_r.csv
triples total: 7641   at 5uM: 5396
lines: 50   drugs complete: 107   pairs: 5350   NA in lines: True
dropped drugs: [('L-Thyroxine (sodium salt pentahydrate)', 46)]
responding: pairs=4593 r=0.5815 sb=0.7353 sqrt_sb=0.8575
all:        pairs=5350 r=0.0812 sb=0.1503 sqrt_sb=0.3876
# rung0_dose_strata.csv, dose 5.0, per_triple
responder 0.5768 / 0.7316   sqrt(SB)= 0.8553
all       0.0808 / 0.1494   sqrt(SB)= 0.3865
```

The leakage exclusions, re-derived through the shipped functions against the committed drug-table
fixture:

```
$ uv run python -  # grid_from_rung0 -> attach_drug_metadata -> sciplex_exposed_pairs
grid: 50 lines x 107 drugs = 5350
exposed pairs found: 2 (('ACH-000681', 'Temsirolimus'), ('ACH-000681', 'Trametinib'))
"Selinexor " in grid drugs: True
```

Cross-references, the plan's file map, the code gates, and the notebooks:

```
$ uv run python -   # every relative link in the four task documents
design.md: ALL RESOLVE   plan.md: ALL RESOLVE   decisions.md: ALL RESOLVE   verification.md: ALL RESOLVE
$ uv run python -   # every path in plan.md's file map
file-map paths checked: 39   missing: []
$ uv run ruff check src tests scripts        All checks passed!
$ uv run ruff format --check src tests scripts   71 files already formatted
$ uv run python -   # notebooks as JSON
verify.ipynb  cells: 24  code: 12  total outputs: 0  non-null exec counts: 0
summary.ipynb cells: 21  code: 10  total outputs: 0  non-null exec counts: 0
```

**Checksums.** [`audit_checksums.json`](audit_checksums.json) records the sha256 and byte length of
all **68** artifacts this audit read, at `55adfb7`. It lists no run output, because none exists.

**Not done here.** Code quality (a separate whole-branch review runs in parallel) and the numbers
(the verification battery is that). This is a documentary diff check only.

## Counts

One canonical count. **197 claims** in scope: **131** from `design.md` (`D1–D131`), **10** from
`docs/SPEC.md`'s rung 1 entry (`S1–S10`), **56** from `plan.md` (`P1–P56`). Each series is
contiguous with no gaps and no repeated number, and every claim below carries exactly one verdict;
both facts were checked by script over this file's own tables rather than counted by eye.

| Verdict | D | S | P | Total |
|---|---|---|---|---|
| ALIGNED | 122 | 9 | 49 | **180** |
| DEVIATION-RECORDED | 4 | 0 | 2 | **6** |
| DRIFT | 4 | 1 | 2 | **7** |
| PENDING-RUN | 1 | 0 | 3 | **4** |
| **Total** | **131** | **10** | **56** | **197** |

- **DRIFT (7):** D26, D108, D113, D131, S6, P17, P18.
- **DEVIATION-RECORDED (6):** D72, D93, D94, D109, P5, P45.
- **PENDING-RUN (4)** — claims whose entire content is a run output: D107, P36, P47, P48.

**Verdicted now, artifact still pending (55).** These carry a real verdict, because the substance of
each is a property of the code or the documents; what the run adds is the realized artifact. They
are exactly the claims named in the [checklist](#pending-run-the-artifact-checklist)'s *Confirms*
column: D7, D11, D24, D25, D26, D28, D29, D42, D49, D55, D56, D57, D58, D61, D63, D64, D65, D66,
D67, D68, D69, D70, D71, D73, D74, D75, D76, D77, D78, D79, D80, D81, D82, D83, D86, D87, D88, D89,
D90, D91, D92, D93, D94, D95, D96, D97, D98, D99, D100, D101, P14, P31, P38, P43, P49.

**Reverse direction:** 69 of 69 changed files accounted; **3 files** are under no claim (R1–R3).

## Clause verdicts — `design.md` (D)

### §1 Question and hypotheses

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D1 | Tahoe-100M measured 379 drugs across 50 cancer cell lines | ALIGNED | `tests/fixtures/tahoe_drug_metadata.csv` holds 379 drug rows (recount); the grid recompute gives 50 lines |
| D2 | H1: Stack's embedding of a line beats (a) the drug's average effect in other lines and (b) the same model given expression | ALIGNED | `heldout/__init__.py:81` `lolo_stack_base_vs_drug_average` H1(a); `:82` vs `expression` H1(b) |
| D3 | H2: the sci-Plex fine-tune beats the released cytokine-tuned Stack | ALIGNED | `heldout/__init__.py:86,91` `stack_drug` vs `stack_cytokine`, hypothesis `H2`, both schemes |
| D4 | Estimates, 95% CIs, p-values and MDEs are reported; nothing passes or fails | ALIGNED | `comparisons.py:46-66` `RedrawSummary` carries estimate/ci_lo/ci_hi/p/sd/mde; `docs/SPEC.md:67` "it has no pass mark" |

### §2 What stays fixed

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D5 | Data is the Tahoe pseudobulk-DE tranche `tahoe100m-pseudobulk-de.v1`, as in rung 0 | ALIGNED | `data/tranches/tahoe100m-pseudobulk-de.v1.json` and `.manifest.txt` present; `heldout_answers.py:92` scans that table |
| D6 | Dose 5 uM only — the most repeated measurements, 5,396 of rung 0's 7,641 | ALIGNED | recomputed: 5,396 of 7,641 at 5 uM; `grid.py:33` `DOSE_UM = 5.0` |
| D7 | 107 drugs × 50 lines = 5,350 pairs, each on at least two plates | ALIGNED | recomputed 50 × 107 = 5,350; rung 0's promoted table holds only splittable (≥2-plate) triples, so every grid pair has ≥2. Realized plate counts arrive with PR-2 |
| D8 | Every drug in every line, so no gaps skew averages; one drug measured in only 46 lines is dropped | ALIGNED | recomputed: the single incomplete drug is `L-Thyroxine (sodium salt pentahydrate)` at 46 lines; `grid.py:82-96` keeps only `count == len(lines)` and raises on an incomplete grid |
| D9 | Answer = each gene's `log2FoldChange` averaged over the pair's plates | ALIGNED | `heldout_answers.py:88` `avg(t.log2FoldChange) AS mean_lfc`, grouped by (line, drug, gene) |
| D10 | Score = per-pair Pearson across genes, averaged over pairs (rung 0's score) | ALIGNED | `scoring.py:135` `masked_rowwise_pearson`, the function moved out of rung 0's `delta_reproducibility.py` |
| D11 | Ceiling = rung 0's promoted 5 uM results limited to these pairs; rung 0 is not re-run | ALIGNED | `grid.py:146-186` filters the promoted per-pair table and reads the promoted dose-strata row unchanged; no rung 0 recomputation in the diff. Written table is PR-1 |
| D12* | Responding genes = `padj < 0.05` on at least one of the pair's plates | ALIGNED | `heldout_answers.py:89` `min(padj) AS min_padj`; `scoring.py:120` selects `answers.responding` |
| D13* | All genes = every measured gene; a pair needs 50 genes; all models use the same pairs | ALIGNED | `scoring.py:36` `MIN_GENES = 50`; `:33` both gene sets; `answers.scoreable` depends on the answer alone, not the model |

<sub>\* D12 and D13 are the two assertions of the design's single "Genes" row, split per `docs/audit.md` ("Split a sentence carrying two assertions into two claims"). The §2 block is D5–D13; §3 resumes at D14.</sub>

### §3 How a line or a drug is hidden

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D14 | Leave one line out (main), 50 rounds; models learn from the other 49 and predict every drug for the hidden line | ALIGNED | `heldout_fit.py:93` `FULL_GRID_LINES = 50`; `rung1_fit.sbatch:38` `N_LOLO_ROUNDS=50`, indexes 0-49 |
| D15 | The hidden line is seen only as it looks with no drug (§4) | ALIGNED | `_fit_lolo` (`heldout_fit.py:340-356`) reads only kernels built from DMSO descriptions; `models.py:45-47` "No function reads the hidden unit's answers" |
| D16 | Leave one drug out (second), 107 rounds; learn from the other 106, predict it in every line | ALIGNED | `heldout_fit.py:93` `FULL_GRID_DRUGS = 107`; `rung1_fit.sbatch:39,52` indexes 50-156 |
| D17 | The hidden drug is seen only as its chemical structure | ALIGNED | `_fit_lodo` (`heldout_fit.py:359-373`) passes only `inputs.tanimoto` for the held drug: `T[held, drugs]` (`models.py:719`) |
| D18 | Settings are chosen inside each round from training data only | ALIGNED | `ridge_lolo`/`ridge_lodo` tune over `_training_indices` (`models.py:215-222`, `380`, `703`); losses are over training entries only |
| D19 | A line and a drug are never hidden together | ALIGNED | the two schemes are disjoint round families (`heldout_fit.py`, `rung1_fit.sbatch:45-52`); no combined scheme exists in `MODELS`/`COMPARISONS` |

### §4 What models are told about lines and drugs

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D20 | Each line is described once, from untreated cells given only DMSO | ALIGNED | `cells.py:39` `DMSO_DRUG = "DMSO_TF"`; `heldout_dmso_cells.py` keeps only that drug |
| D21 | Expression = the line's average log CPM over Stack's 15,012 genes | ALIGNED | `heldout_dmso_cells.py:90` `GENE_PANEL_SIZE = 15012`; `cells.py:148-178` `pseudobulk_log_cpm` returns log2(CPM+1) |
| D22 | PCA = the few directions in which the 50 lines' expression differs most | ALIGNED | `descriptions.py:91-111` `pca_components`, SVD with a fixed per-component sign |
| D23 | NMF = expression rebuilt as a mix of a few gene programs | ALIGNED | `descriptions.py:123-134` `nmf_components`, `init="nndsvda"`, seeded |
| D24 | Stack base = mean Stack embedding from the pretrained `bc_large.ckpt` | ALIGNED | `leakage.py:43` `("stack_base", "bc_large.ckpt")`; `heldout_embed.py:3` |
| D25 | Stack cytokine = the same from the released `bc_large_aligned.ckpt` | ALIGNED | `leakage.py:44` |
| D26 | Stack drug = the same from our sci-Plex fine-tune, archive file `epoch=5-val_loss=6.1078.ckpt` | **DRIFT** | `leakage.py:45` records `"epoch=5-val_loss=6.1078.ckpt"`, matching the design; but the only place in the tree that says which file is loaded names a different one — `heldout_embed.py:41`, `finetuned-epoch=5-val_loss=6.1078.ckpt`. The provenance record and the job disagree about the checkpoint's filename, and nothing records which is right |
| D27 | Random stand-in = random numbers of the same size, drawn once per line with a fixed seed | ALIGNED | `descriptions.py:152-155` `random_stand_in`; `:49-56` `RANDOM_SEEDS`, one per description |
| D28 | Every description uses the same cells: up to 1,000 DMSO cells per line | ALIGNED | `cells.py:107-145` `select_cells(..., per_line=1000)`; `heldout_dmso_cells.py:661` `--per-line` default 1000. Realized counts are PR-3 |
| D29 | Cells are spread evenly over the line's plates | ALIGNED | `cells.py:130` quota `ceil(per_line / plates_per_line[line])` per (line, plate) |
| D30 | Cells are picked with a fixed seed | ALIGNED | `cells.py:58-79` `cell_keys` is splitmix64 of the packed shard position, seed default 0 — independent of scan order |
| D31 | Stack gets raw counts, as it was trained, in groups from one line only | ALIGNED | `heldout_embed.py` calls `get_latent_representation` on one `cells/line_{i}.h5ad` at a time, so each group holds one line; `DATA.md` records raw counts over the panel |
| D32 | Descriptions are rescaled across all 50 lines, and PCA and NMF use all 50 | ALIGNED | `descriptions.py:62-88` standardizes columns across rows (lines); `pca_components`/`nmf_components` are given the full 50-line matrix |
| D33 | Descriptions hold no drug responses, so this reveals nothing about a hidden answer | ALIGNED | every description is built from `DMSO_TF` cells only (`cells.py:39`, `heldout_dmso_cells.py`); no answer array enters `descriptions.py`. (The plan's *named test* for this is missing — see P15) |
| D34 | A drug is described by a Morgan fingerprint from Tahoe's drug table | ALIGNED | `chemistry.py:28-53` `morgan_fingerprints(radius=2, n_bits=1024)` over the table's `canonical_smiles` |
| D35 | Two drugs' similarity is the share of groups they have in common (Tanimoto) | ALIGNED | `chemistry.py:56-72` `tanimoto`, `inter / (a + b - inter)` |

### §5 The models

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D36 | Drug average (hiding lines) = the drug's average change in the 49 training lines, no line description | ALIGNED | `models.py:192-208` `drug_average(delta0, train)`; `heldout_fit.py:347-350` the `reference` branch under LOLO |
| D37 | Chemistry only (hiding drugs) = each line's average over training drugs, adjusted using chemically similar drugs, no line description | ALIGNED | `heldout_fit.py:367-369`: the `reference` branch under LODO is `ridge_lodo` with an all-zero line kernel; `models.py:32-33` documents exactly that |
| D38 | Ridge regression for every description — expression, PCA, NMF, three Stack versions, and each random stand-in | ALIGNED | `heldout/__init__.py:58-70` declares six ridge models and six `random_*` stand-ins; `heldout_fit.py:280-301` builds a kernel or candidate set for each |
| D39 | One method for all, so only the description changes | ALIGNED | `_fit_lolo`/`_fit_lodo` dispatch on `spec.kind`, not on the description; every ridge model goes through the same two functions |
| D40 | Hiding a line: learns how the 49 lines' departures from the drug average follow their descriptions, predicts the hidden line's departure, adds the average back | ALIGNED | `models.py:352-392` `ridge_lolo`: `centre = _line_mean(delta0, lines)`, prediction is `centre + weights @ departures` (`:343-349`) |
| D41 | Hiding a drug: learns from all training pairs at once, letting chemically similar drugs share effects on average and in lines with similar descriptions | ALIGNED | `models.py:734-761` `ridge_lodo`, pair kernel `T[d,d'] * (1 + K_line[l,l'])` |
| D42 | The penalty and the number of PCA/NMF components (2, 5, 10, 15 or 20) are chosen by leaving out one training line (or drug) at a time | ALIGNED | `models.py:72` `LAMBDAS = logspace(-3, 3, 13)`; `heldout_fit.py:84` and `heldout_descriptions.py:66` `(2, 5, 10, 15, 20)`; `ridge_lolo_k`/`ridge_lodo_k` choose k and λ jointly by exact leave-one-out loss. Realized choices are PR-6 |
| D43 | Nearest lines (hiding lines) = the average change in the k training lines whose expression looks most like the hidden line | ALIGNED | `models.py:460-503` `nearest_lines_lolo`; `heldout_fit.py:311` similarity is built from `descriptions["expression"]` |
| D44 | k (3, 5, 10 or 20) is chosen the same way | ALIGNED | `models.py:465` `ks: Sequence[int] = (3, 5, 10, 20)`, chosen by `_leave_one_out_losses` |
| D45 | Untested genes count as zero change in training and are never scored | ALIGNED | `heldout_fit.py:307` `delta0 = np.nan_to_num(answers.delta, nan=0.0)`; `scoring.py:120-122` selects only `np.isfinite(measured)` |
| D46 | The drug average holds everything about a pair that does not depend on the line, so a model beats it only by using what it was told about the line | ALIGNED | rationale consistent with `drug_average` being line-blind by construction (`models.py:192-208`, a mean over training lines only) |
| D47 | Each random stand-in goes through the same fit, so a description that beats its stand-in gains from the description | ALIGNED | `heldout_fit.py:343-345` both ridge branches look matrices up **by model id**, and a tuned description's stand-in gets the same candidate counts (`:291-301`, ruling 35) |
| D48 | For hidden drugs, chemistry only plays this role | ALIGNED | `heldout/__init__.py:87` the LODO H1(a) comparison is `stack_base` vs `chemistry_only` |

### §6 Scoring, and why the ceiling is a square root

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D49 | Each model gets its average score over pairs with a CI, also as a fraction of √SB | ALIGNED | `heldout_combine.py:531-566` `model_summary_table` writes the mean, its redraw interval and `fraction_of_ceiling`; `scoring.py:140-144`. Numbers are PR-8 |
| D50 | A measurement is X = T + e, R = var(T)/var(X); two measurements correlate at R, the true change at √R | ALIGNED | derivation; implemented as the score control — `controls.py:97-115` `planted_reliability_pool`, asserted by `test_score_square_root_known_answer` |
| D51 | Dividing by R would let a perfect model score over 100% | ALIGNED | consistent with D50; `fraction_of_ceiling` divides by `sqrt_sb`, not by `sb` (`scoring.py:141`) |
| D52 | The archive's rung 1 saw models at 0.28–0.32 beat a "ceiling" of 0.109; against √0.197 = 0.44 they reach 62–72% | ALIGNED | arithmetic checks: √0.197 = 0.4438; 0.28/0.4438 = 63%, 0.32/0.4438 = 72%. The archive figures are claims about branch `worktree-modular-harness-core` (present locally and on origin), not about this tree |
| D53 | Rung 1 predicts the average over all plates, so R is rung 0's full-data (Spearman-Brown) reliability | ALIGNED | `grid.py:171` `sb = spearman_brown(split_half_r)`; `statistics.py:131` |
| D54 | The values come from rung 0's promoted per-pair table, filtered to our pairs | ALIGNED | `grid.py:155-158` filters the promoted table to the grid's lines and drugs at 5 uM |
| D55 | Responding row: 4,593 pairs, r 0.5815, SB 0.7353, √SB 0.8575 | ALIGNED | recomputed exactly: `pairs=4593 r=0.5815 sb=0.7353 sqrt_sb=0.8575` |
| D56 | All-genes row: 5,350 pairs, r 0.0812, SB 0.1503, √SB 0.3876 | ALIGNED | recomputed exactly: `pairs=5350 r=0.0812 sb=0.1503 sqrt_sb=0.3876` |
| D57 | Rung 0's promoted 5 uM responding row is 0.5768 / 0.7316 / 0.8553 | ALIGNED | recomputed from `rung0_dose_strata.csv`: 0.5768 / 0.7316, √SB 0.8553 |
| D58 | Rung 0's promoted 5 uM all-genes row is 0.0808 / 0.1494 / 0.3865 | ALIGNED | recomputed: 0.0808 / 0.1494, √SB 0.3865 |
| D59 | Known limit 1 — averaging over pairs before the square root makes the ceiling a little high | ALIGNED | stated as a limit and carried to the reader: `summary.ipynb` contains "Averaging order" |
| D60 | Known limit 2 — rung 0 picked responding genes from half the plates, rung 1 from all of them | ALIGNED | `heldout_answers.py:89` takes `min(padj)` over all the pair's plates, against rung 0's half-based selection; carried in `summary.ipynb` |
| D61 | Known limit 3 — 50 pairs have three plates, so the fraction reads a little high for them | ALIGNED | recomputed: at 5 uM exactly **50** grid pairs have `n_plates_even == False`, i.e. unequal split halves, which is an odd plate count (`delta_reproducibility.py:269`); carried in `summary.ipynb` ("three plates"). The exact multiplicity per pair arrives with PR-2 |
| D62 | Known limit 4 — all 50 lines share wells, so a well's lab error reaches hidden and training lines alike | ALIGNED | stated as an unmeasured limit; carried in `summary.ipynb` ("Shared wells") |

### §7 Comparisons and statistics

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D63 | Each hypothesis is a head-to-head comparison on the same pairs, on responding genes | ALIGNED | `heldout_combine.py:645-655` Holm's families are the declared comparisons on `gene_set == "responding"`; every model is scored on one shared population (`scoring.py`, invariant 2) |
| D64 | LOLO family (6): Stack base − drug average, for H1(a) | ALIGNED | `heldout/__init__.py:81` |
| D65 | LOLO: Stack base − expression, − PCA, − NMF, − nearest lines, for H1(b) | ALIGNED | `heldout/__init__.py:82-85` |
| D66 | LOLO: Stack drug − Stack cytokine, for H2 | ALIGNED | `heldout/__init__.py:86` |
| D67 | LODO family (5): Stack base − chemistry only, for H1(a) | ALIGNED | `heldout/__init__.py:87` |
| D68 | LODO: Stack base − expression, − PCA, − NMF, for H1(b) | ALIGNED | `heldout/__init__.py:88-90` |
| D69 | LODO: Stack drug − Stack cytokine, for H2 | ALIGNED | `heldout/__init__.py:91`; `test_lolo_and_lodo_comparison_counts` pins 6 and 5 |
| D70 | Redraw the hidden units with repeats (the 50 lines, or the 107 drugs) 2,000 times | ALIGNED | `comparisons.py:38` `N_DRAWS = 2000`; `:85-116` multinomial unit weights over `n_units` |
| D71 | Confidence interval = the middle 95% of redraws | ALIGNED | `comparisons.py:183` `np.percentile(finite, [2.5, 97.5])` |
| D72 | p-value = twice the share of redraws past zero | **DEVIATION-RECORDED** | `comparisons.py:184-191` uses the +1 resampling estimator, `min(1, 2·min((1+#{≤0})/(1+B), (1+#{≥0})/(1+B)))`. Recorded in `decisions.md` (2026-09-11, ruling 42) with the reason and the note that the battery checks the code's convention |
| D73 | MDE = 2.8 × the redraws' standard deviation (5% false positives, 80% power) | ALIGNED | `comparisons.py:43` `MDE_FACTOR = 2.8`; `:193` `mde = MDE_FACTOR * sd` |
| D74 | p-values are adjusted by Holm's method within each test | ALIGNED | `heldout_combine.py:646-655`: one family per scheme; `comparisons.py:235-250` `holm` |
| D75 | Pairs share lines and drugs, so lines and drugs are also redrawn together once, and how much wider intervals get is reported | ALIGNED | `comparisons.py:119-148` `two_way_estimates`; `heldout_combine.py:639-640` writes both `design_effect` and `width_ratio`. `decisions.md` (2026-09-11) records that both are reported |
| D76 | Also reported unadjusted: the same comparisons on all genes | ALIGNED | `heldout_combine.py:612-613` loops both gene sets; `:598-600` all-genes rows keep `p_holm` empty |
| D77 | Also reported unadjusted: each description against its random stand-in | ALIGNED | `heldout_redraws.py:113-126` `stand_in_contrasts` pairs each ridge description with `random_<id>` on each scheme it runs; `heldout_combine.py:520-524` appends them unadjusted |
| D78 | Also reported: each model's average score and fraction of the ceiling | ALIGNED | `heldout_combine.py:176`, `:531-566` `fraction_of_ceiling` column in `rung1_model_summary.csv` |
| D79 | The sci-Plex fine-tune saw A549 (ACH-000681) with five Tahoe drugs (CIDs 6918289, 11626560, 104741, 11707110, 3385) | ALIGNED | `grid.py:37` `SCIPLEX_CIDS` is exactly those five; `:40` `SCIPLEX_LINE = "ACH-000681"` |
| D80 | Those pairs are removed for every model, and the count is reported | ALIGNED | recomputed through the shipped functions: 2 grid pairs, `(ACH-000681, Temsirolimus)` and `(ACH-000681, Trametinib)` — matching `plan.md:29`; `leakage.py:81` reports `n_exposed_pairs`; `scoring.py` drops excluded pairs for all models |
| D81 | Stack's pretraining may include untreated cells of these lines: input, not answer; no Stack saw Tahoe's treated cells | ALIGNED | `leakage.py:52-55` `PRETRAINING_NOTE`, verbatim; `:73` `saw_tahoe_treated_cells=False` |
| D82 | Each model version's exposure is written to a `LeakageProfile` record | ALIGNED | `leakage.py:58-133`, one profile per `STACK_CHECKPOINTS` entry, refusing to write a record it cannot stand behind (`:99-111`). Written file is PR-11 |

### §8 Figures, controls and power

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D83 | Each step has a positive control (a planted answer the real code must find) and a negative control | ALIGNED | `heldout/controls.py` builds the planted grids; `tests/test_rung1_controls.py` and `tests/test_heldout_descriptions.py` implement one pair per step |
| D84 | Each figure is drawn from a table the run writes, with its control beside it | ALIGNED | `heldout_combine.py:1196-1200`: each `fig_*` is called with the tables just written, including its `rung1_control_*` frame |
| D85 | Tolerances go in `plan.md`, from each test's spread by chance | ALIGNED | `plan.md:104-107` invariant 9: 3 Monte Carlo standard errors; rates over 200 repetitions with a binomial 99% interval |
| D86 | build positive: a half-cell description best matches the line's other half, for more lines than the negative control's 99th percentile | ALIGNED | `tests/test_heldout_descriptions.py::test_build_identity_recovered`; `descriptions.py:193-198` `identity_match` |
| D87 | build negative: with line labels shuffled, matches fall to chance (1 in 50) | ALIGNED | `tests/test_heldout_descriptions.py::test_build_shuffled_identity_is_chance`; `descriptions.py:263-294` `shuffled_identity_null` |
| D88 | build check: the drug fine-tune's weights differ from `bc_large.ckpt` | ALIGNED | `heldout_embed.py:378` `encoders_differ` over shared encoder keys, writing `rung1_weights_check.json` and failing if identical. Artifact is PR-4 |
| D89 | build figures: cells per line and plate; half-versus-half match grids per description, shuffled grid beside | ALIGNED | `figures.py:182` `fig_build`; panels `:257` "(a) cells per line and plate", `:292` the match panel, `:320` per-description grids, with the shuffled control. PNG is PR-10 |
| D90 | split positive: a signal planted only in one unit's own responses is found by a broken split that leaves the hidden unit in training | ALIGNED | `tests/test_rung1_controls.py::test_split_leaky_recovers_signature`; `controls.py:255-297` `leaky_lolo_prediction`/`leaky_lodo_prediction` |
| D91 | split negative: the real splits score it at zero, and a test confirms no hidden value reaches training | ALIGNED | `::test_split_shipped_does_not`; plus `test_lolo_ignores_held_out_answers` and `test_lodo_ignores_held_out_answers` (`tests/test_heldout_models.py`) |
| D92 | split figures: the 50 × 107 grid of rounds; the planted signal under the broken and the real split | ALIGNED | `figures.py:337` `fig_split(pair_scores, split, lines, drugs, out)`; panels `:383` (the grid) and `:422` (the control). PNG is PR-10 |
| D93 | fit positive: on a synthetic grid of the screen's size and noise, a line-specific response planted at twice its MDE is recovered by the matching description and not by its stand-in | **DEVIATION-RECORDED** | `decisions.md`, "design.md (execution)", 2026-09-11: "planted at twice their MDE is unattainable as written" — measured, and replaced by a fixed plant (strength 0.3) at R = 0.7353 with the oracle's gain asserted ≥ 2 × its own MDE. `controls.py:333` `STRENGTH = 0.3`, `:326` `R_RESPONDING = 0.7353` |
| D94 | fit negative: with nothing planted, no model beats the drug average by more than its MDE | **DEVIATION-RECORDED** | same entry: the null is read one-sided (gain ≤ MDE), and the 2026-09-11 amendment makes the line-hidden stand-in check one-sided too, with the reason. `controls.py:567` `fit_null_control` |
| D95 | fit figures: settings chosen per round; planted gain found, beside each model's real score | ALIGNED | `figures.py:438` `fig_fit`; panels `:487` "(a) the penalty each round chose", `:510` "(b) the component count each round chose", `:541` the control, `:568` "(d) each model's real score". PNG is PR-10 |
| D96 | score positive: at the §6 reliabilities the true change scores √R and a second noisy copy scores R, through the real scoring code | ALIGNED | `tests/test_rung1_controls.py::test_score_square_root_known_answer`; `controls.py:326-327` `R_RESPONDING = 0.7353`, `R_ALL = 0.1503` — the design's own SB values |
| D97 | score negative: a prediction unrelated to the answer scores zero | ALIGNED | `::test_score_independent_prediction_is_zero`; `controls.py:348` `SCORE_UNRELATED_SEEDS` |
| D98 | score figures: each model's scores with √SB and rung 0's r marked; the √R test beside it | ALIGNED | `figures.py:591` `fig_score(summary, ceiling, score, out)` — the ceiling table supplies √SB and the promoted r; panels `:679`, control `:717`. PNG is PR-10 |
| D99 | null positive: a comparison planted exactly at its MDE is detected about 80% of the time | ALIGNED | `::test_null_detection_at_mde`; `controls.py:354` `DETECTION_RATE = 0.80` over `:352` `REPETITIONS = 200` |
| D100 | null negative: with nothing planted, detections happen at most 5% of the time | ALIGNED | `::test_null_false_positive_rate`; `controls.py:353` `ALPHA = 0.05` |
| D101 | null figures: every comparison with its interval and MDE, both gene sets; redraws for H1(a) and H2 | ALIGNED | `figures.py:758` `fig_null`; `:818` "(a) every comparison with its interval and MDE, both gene sets"; `:733-756` `null_panel_comparisons` selects the H1(a) and H2 panels. PNG is PR-10 |

### §9 Running on Alpine

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D102 | Heavy steps run on Alpine only through `scripts/alpine/ralpine`, never plain ssh or file copies | ALIGNED | every job is submitted through `ralpine submit` in `submit_rung1_chain.sh`; `tests/test_ralpine_boundary.py` enforces the boundary |
| D103 | Push the branch, `ralpine switch` once, then `ralpine update` and `ralpine submit`; first confirm inputs with `ralpine ls`/`du` and pass the synthetic tests locally | ALIGNED | `ralpine:227,229,246,298,346` provide `ls`, `du`, `switch`, `update`, `submit`; `verification.md` records the local suite and the switch having run on Alpine |
| D104 | Run everything possible in parallel, as job arrays | ALIGNED | five of the ten stage jobs are arrays (`answers` 0-7, `dmso_cells` 0-63%4, `embed` 0-2, `fit` 0-156, `redraws` 0-7) |
| D105 | Chain stages with `--dependency=afterok:<job id>`, and confirm each with `ralpine jobinfo <id>` | ALIGNED | `submit_rung1_chain.sh:82-90` `check_dependency` reads the dependency back with `ralpine jobinfo` and aborts if it is absent; used at every link (`:110,122,129,147,152,160,165,174,179`) |
| D106 | Resubmitting: a task whose output already exists skips, so a failed task reruns alone | ALIGNED | `records.py:48-60` `is_done` requires the file *and* a matching sha256; every stage's docstring states the skip |
| D107 | Logs: pull every log into `results/rung1-held-out-prediction/logs/` | **PENDING-RUN** | the directory does not exist; no job has run. Needs: the pulled log files, and `verification.md`'s pointer resolving |
| D108 | Stage 1 (DMSO cells and drug table) is an array over source files, at most 4 at once, waiting for nothing | **DRIFT** | the `%4` half holds (`rung1_dmso_cells.sbatch:11`, `--array=0-63%4`), but the waits-for half does not: `submit_rung1_chain.sh:160` submits the DMSO array with `--dependency=afterok:$GRID`. §9's stage table has **no grid stage at all**, although the tree has `rung1_grid.sbatch` and both data stages depend on it. `plan.md:312` lists the job; nothing records the table's omission |
| D109 | Stage 2 (answers, 5 uM, 107 drugs) reads rung 0's 32 cached gene slices, re-running rung 0's slice array if scratch was purged | **DEVIATION-RECORDED** | the tree runs a fresh filtered DuckDB scan in 8 parts (`heldout_answers.py:80-92`, `rung1_answers.sbatch:20`, `N_ANSWER_PARTS=8`). `decisions.md` (2026-09-11, under the `## plan.md` heading) records it: "The answers are scanned, not read from rung 0's cache", with the reason (the cache keeps only genes with a fold change in *both* halves, which would narrow an approved definition). See observation O1 on where the entry is filed |
| D110 | Stage 3 (Stack embeddings) is a 3-task GPU array — base, cytokine, drug — waiting for stage 1 | ALIGNED | `rung1_embed.sbatch:12` `--array=0-2  # 0 base, 1 cytokine, 2 drug`; `submit_rung1_chain.sh:174` waits on the DMSO combine |
| D111 | Stage 4 (expression, PCA, NMF, random descriptions) is one small job waiting for stage 1 | ALIGNED | `rung1_descriptions.sbatch` (no `--array`); `submit_rung1_chain.sh:179` waits on the DMSO combine |
| D112 | Stage 5 (model fits) is a 157-task array, one per round (50 lines + 107 drugs), every model per task | ALIGNED | `rung1_fit.sbatch:21` `--array=0-156`, `:38-52` maps 0-49 to `lolo` and 50-156 to `lodo`; `heldout_fit.py` fits every model declared for the scheme |
| D113 | Stage 5 waits for stages 2, 3 and 4 | **DRIFT** | it does not. `submit_rung1_chain.sh:25-26`: "The two stages are separate chains: `--stage fit` does NOT wait for `--stage data` unless you pass `--after <job id>`", and `:98-100` says so again. `verification.md:38,50-54` documents the gap and tells the operator to sequence it by hand — but `decisions.md` carries no entry, so this is drift, not a recorded deviation |
| D114 | Stage 6 (redraws) is an array of 8 tasks × 250 redraws, per test, waiting for stage 5 | ALIGNED | `rung1_redraws.sbatch:17` `--array=0-7  # N_BLOCKS = 8 blocks of DRAWS_PER_BLOCK = 250`; `comparisons.py:39-40`; `submit_rung1_chain.sh:122` waits on the whole fit array |
| D115 | Stage 7 (tables and figures) is one job waiting for stage 6 | ALIGNED | `rung1_combine.sbatch` (no array); `submit_rung1_chain.sh:129` waits on the whole redraw array |
| D116 | `acpu` is the general CPU partition, `--qos=cpu-normal` (`amilan` retired 2026-08-24) | ALIGNED | every CPU job carries `--partition=acpu --qos=cpu-normal` (grid, answers, answers_combine, dmso_cells, dmso_combine, descriptions, fit, redraws, combine) |
| D117 | `acpu` had 1,353 jobs waiting, 914 ahead of this account; `amem` had eight idle nodes; rung 0's slices finished in 2.5–2.8 h on three named nodes | ALIGNED | historical observations of 2026-09-10/11, sourced in `decisions.md` (review round 1) to rung 0's `decisions.md`, its `verification.md` job table and its job logs. Not re-derivable from this tree, and not re-derived |
| D118 | If `acpu` is deep and `amem` is idle, pack stage 5 onto `amem` (`--qos=mem-normal`, ≥ 256 GB) | ALIGNED | a conditional instruction for submission time, repeated at `plan.md:324-326`; no `amem` job is required in the tree until the queue check says so |
| D119 | GPU work moved from `aa100` to `ah200`, with `--qos=gpu-normal --gres=gpu:h200:1` | ALIGNED | `rung1_embed.sbatch:3-5` exactly that, with `:18` recording the `aa100` fallback |
| D120 | Before submitting, check the queues with `ralpine run squeue -p acpu -t PENDING` and `ralpine run sinfo -p amem` | ALIGNED | `ralpine:329` provides `run`; the instruction is repeated at `plan.md:324` |
| D121 | Alpine charges memory per core (3,840 MB), so ask for memory in whole cores | ALIGNED | every CPU job's request is a whole multiple: 15G = 4 × 3,840; 75G = 20 ×; 60G = 16 ×; 30G = 8 ×; 7,680M = 2 × |
| D122 | Scratch reads were 3 times slower on 2026-09-10, so give time limits slack | ALIGNED | a stated observation with slack in the job walltimes (2–6 h per stage, `verification.md` table) |

### §10 Commits and the pull request

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D123 | Work in the worktree `.claude/worktrees/rung1-held-out-prediction` on `rung1-held-out-prediction-work`, committing and pushing as needed | ALIGNED | `git branch` shows `* rung1-held-out-prediction-work`; `remotes/origin/rung1-held-out-prediction-work` exists, so the branch has been pushed |
| D124 | Before the pull request, the final files are recommitted as five topic commits on `rung1-held-out-prediction`, from the same base | ALIGNED | not yet due — the task stops at gate 2 (`plan.md:357`, brief "do not rebuild the branch into topic commits"). The target branch exists locally |
| D125 | Results come later; until then no run output enters git | ALIGNED | the branch diff carries no result table, figure or log; `git status` shows the untracked scratch left untracked |
| D126 | `git diff rung1-held-out-prediction-work rung1-held-out-prediction` must be empty before pushing, and the PR comes from `rung1-held-out-prediction` | ALIGNED | not yet due (no PR at gate 2); the check is a pre-push condition, not a current tree property |

### §11 Earlier evidence (archive branch)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D127 | Archive rung 1 (doses pooled, 1,508 pairs, lines hidden): drug average 0.315, PCA 0.319, NMF 0.313, nearest lines 0.276; no expression model beat the drug average | ALIGNED | the cited branch `worktree-modular-harness-core` exists locally and on origin (`git branch -a`). These are claims about that branch's results, not about this tree; the citation resolves and the numbers were not re-derived |
| D128 | Stack generating responses scored near zero (0.012 cytokine, 0.021 drug), even with the hidden line leaked in | ALIGNED | as D127 |
| D129 | For cell survival only Stack base showed a line-specific signal (0.119), but a line-blind model reached 0.118, blamed on 17 missing pairs; hence the complete grid | ALIGNED | as D127; and the complete grid it motivates is realized (recomputed 50 × 107 with no gaps) |

### §12 Not in this task, and what changes on approval

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D130 | Not in this task: Stack generating responses; the 0.05 and 0.5 uM doses; re-running rung 0; hiding a line and a drug together | ALIGNED | no generation code in `src/fmharness/heldout/` (the only `generat*` hits are `heldout_embed.py:187,378`, which *strip* the generation head); `grid.py:33` fixes 5.0 uM; `delta_reproducibility.py`'s only change is the import move; no combined scheme exists |
| D131 | On approval, `docs/SPEC.md`'s rung 1 entry is rewritten to match §1–§7 (proposed text in `decisions.md`), and STATE and README are updated to match | **DRIFT** | STATE and README were updated (`docs/STATE.md:20`, `README.md:73-75`), and SPEC was rewritten — but **not to the proposed text `decisions.md` carries**, and nothing records the change. Field-by-field (normalized for line wrapping): *Question* identical; *Adds* gained ", not by what the split assumes"; *Measure* and *Reports* were substantially expanded; *Why it matters* was replaced by a differently-worded *How it contextualises the rest*; a *Tasks* line was added. The document says the proposed text is what gets applied; something else was applied |

## Clause verdicts — `docs/SPEC.md`, rung 1 entry (S)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| S1 | Heading: "can a model predict a cell line or a drug it has not seen?" | ALIGNED | `docs/SPEC.md:62`; matches `design.md:1`'s framing and `README.md:15`'s ladder row |
| S2 | Question: given a line or a drug left out of training, can a model predict the expression change? | ALIGNED | `docs/SPEC.md:64`; realized as the two schemes (D14, D16) |
| S3 | Adds one hidden thing at a time, a line or a drug; same lab method, cells and dose as rung 0 | ALIGNED | `docs/SPEC.md:65`; one hidden unit per round, dose fixed at 5 uM (`grid.py:33`) |
| S4 | What counts as unseen is set by each model's leakage record, not by what the split assumes | ALIGNED | `leakage.py:58-133` `LeakageProfile` per model version, and the excluded pairs are derived from it (`:106-111` refuses a record that disagrees with the grid) |
| S5 | Measure: rung 0's score, at one dose, on a complete grid, for responding genes and for all genes | ALIGNED | `scoring.py:33` both gene sets; recomputed grid is complete (50 × 107, no gaps) |
| S6 | Head-to-head comparisons on the same pairs, each with a CI, p-value and MDE, including **every line description against the line-blind reference** and against a random stand-in of the same size | **DRIFT** | the stand-in half holds — `heldout_redraws.py:113-126` contrasts every ridge description with its own `random_<id>` on both schemes. The line-blind half does not: `heldout_combine.py:504-524` computes exactly the 11 declared comparisons plus the stand-in contrasts, and only `stack_base` is compared to the line-blind reference (`__init__.py:81,87`). Expression, PCA, NMF and nearest lines are compared to `stack_base`, never to the drug average or chemistry only. They appear in `rung1_model_summary.csv` as separate mean scores, which is not a head-to-head contrast with a CI, p and MDE |
| S7 | Each model's score is also given as a fraction of √SB, rung 0's full-data reliability on the same pairs | ALIGNED | `heldout_combine.py:176`, `:564` `fraction_of_ceiling`; `grid.py:180` `sqrt_sb` |
| S8 | The hypotheses under test are named in the task's design | ALIGNED | `design.md:13-18` names H1 and H2; `__init__.py:21` types them and every comparison carries one |
| S9 | The rung reports results; it has no pass mark | ALIGNED | `docs/SPEC.md:67`; no threshold appears in `comparisons.py` or `heldout_combine.py`; `design.md:20` "Nothing passes or fails" |
| S10 | Tasks: the rung 1 design, OPEN, design approved 2026-09-11 | ALIGNED | `docs/SPEC.md:69`; the link resolves (cross-reference check); `design.md:3` carries the same status and date |

## Clause verdicts — `plan.md` (P)

### Global constraints

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| P1 | Dose 5.0 uM only | ALIGNED | `grid.py:33` |
| P2 | Grid 107 × 50 = 5,350: drugs with a replicated 5 uM triple in all 50 lines of rung 0's promoted per-pair table | ALIGNED | recomputed exactly from that file |
| P3 | Answer = mean `log2FoldChange` over the pair's plates; responding = `padj < 0.05` on at least one plate; all = every gene with a finite answer | ALIGNED | `heldout_answers.py:88-91` |
| P4 | A pair is scored on a gene set when ≥ 50 genes qualify | ALIGNED | `scoring.py:36` `MIN_GENES = 50`; `test_scoreable_boundary_49_versus_50_responding_genes` |
| P5 | Comparisons: the §7 table (6 LOLO, 5 LODO), responding genes, 2,000 redraws, CI 2.5/97.5, two-sided p, MDE 2.8 × SD, Holm within each test, two-way design effect | **DEVIATION-RECORDED** | every element holds (see D63–D75) except the p-value convention, which is ruling 42's recorded +1 estimator |
| P6 | Line descriptions: expression over 15,012 genes standardized, PCA, NMF k ∈ {2,5,10,15,20}, three Stack versions, a random stand-in of the same width, up to 1,000 plate-balanced DMSO cells with a fixed seed | ALIGNED | `heldout_dmso_cells.py:90`; `heldout_descriptions.py:66`; `descriptions.py:49-59,152`; `cells.py:107-145` |
| P7 | Models: drug average (LOLO), chemistry only (LODO), ridge for every description (both schemes), nearest lines (LOLO, k ∈ {3,5,10,20}); settings chosen inside the round | ALIGNED | `__init__.py:53-72`; `models.py:465`; `heldout_fit.py:340-373` |
| P8 | Leakage: pairs (ACH-000681, Trametinib) and (ACH-000681, Temsirolimus) are removed for every model | ALIGNED | recomputed through the shipped functions — exactly those two pairs |
| P9 | Ceiling √SB from the promoted per-pair table: responding 0.8575, all 0.3876 | ALIGNED | recomputed: 0.8575 and 0.3876 |
| P10 | Run outputs never enter git; stage files by name, never `git add -A` | ALIGNED | the branch diff carries no run output; this audit's own commit is path-limited |
| P11 | New code passes `ruff check`, `ruff format --check` and `pyright` strict over `src` and `tests` | ALIGNED | re-run here: `All checks passed!` and `71 files already formatted`. `pyproject.toml:61` sets strict pyright over `src`, `tests` and `scripts/verify_rung1.py`; `verification.md:70-71` records 0 errors |

### File map and data contracts

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| P12 | Every path the file map names exists and holds the responsibility described | ALIGNED | script check: 39 file-map paths, 0 missing |
| P13 | Scratch root `RUNG1_CACHE=/scratch/alpine/$USER/rung1_cache`; result tables in `OUT_DIR=docs/tasks/rung1-held-out-prediction` (untracked) | ALIGNED | `rung1_env.sh:29-30`, exactly those defaults; no `OUT_DIR` table is tracked |
| P14 | Every file a stage writes is followed by a `<name>.done.json` holding its sha256, so a rerun skips finished work and the audit can pin bytes | ALIGNED | `records.py:31-45` `write_record`, `:48-60` `is_done`; used by grid, answers, dmso_cells, embed, redraws, combine and fetch_metadata. Realized records are PR-6/PR-7 |
| P15 | The stage-by-stage data contracts (the written files and their columns) | ALIGNED | `heldout_fit.py:99` `SETTINGS_COLUMNS = ("model","lambda","k","loss_min","lambda_at_edge")` matches the contract; `scoring.py:39` `SCORE_COLUMNS`; `heldout_combine.py:145-200` the result-table columns; `grid.py:199-207` the restriction record's fields |

### Invariants

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| P16 | Invariant 1 — no held-out answer reaches fitting or tuning; tests `test_lolo_ignores_held_out_answers`, `test_lodo_ignores_held_out_answers` | ALIGNED | both exist in `tests/test_heldout_models.py` |
| P17 | Invariant 2 — every model is scored on the same pairs; test `test_scoreable_pairs_do_not_depend_on_the_model` | **DRIFT** | the property holds (`scoring.py:63-71`, `answers.scoreable` is a function of the answer and the exclusions only), but **the named test does not exist**. Repo-wide search finds the string only in `plan.md:89`. The nearest implemented tests are `test_scoreable_boundary_49_versus_50_responding_genes` and `test_excluded_pairs_are_false_even_when_otherwise_scoreable` (`tests/test_heldout_answers.py:285,308`), neither of which is the named invariant test |
| P18 | Invariant 3 — descriptions carry no drug response, built only from DMSO cells; test `test_descriptions_read_only_dmso_cells` | **DRIFT** | the property holds (D33), but **the named test does not exist**. Repo-wide search finds the string only in `plan.md:90`. `tests/test_heldout_cells.py:490` `test_dmso_cells_end_to_end` is the closest, and it is not the named test |
| P19 | Invariant 4 — determinism; tests `test_cell_selection_is_independent_of_shard_order`, `test_redraws_are_seeded` | ALIGNED | `tests/test_heldout_cells.py` and `tests/test_heldout_scoring.py:459` |
| P20 | Invariant 5 — ceiling values reproduce the design table; test `test_ceiling_table_matches_design` | ALIGNED | `tests/test_heldout_grid.py`; independently recomputed here to the same four values |
| P21 | Invariant 6 — tuning is closed-form and exact, each inner fit re-estimating the round's reference; tests `test_ridge_lolo_closed_form_loo_equals_refits`, `test_ridge_lodo_block_loo_equals_refits` | ALIGNED | both exist in `tests/test_heldout_models.py`; `models.py:282-325` and `:526-569` implement the re-estimation, and the plan text already carries the corrected position (the reversal is recorded in `decisions.md`) |
| P22 | Invariant 7 — the linear kernel is scaled to a mean diagonal of 1; tuning losses are MSE over tested entries only | ALIGNED | `descriptions.py:158-170` divides by the mean diagonal and raises on an all-zero kernel; `models.py:263-269` masks by `tested` before summing |
| P23 | Invariant 8 — staged equals one process; test `test_staged_run_equals_one_process` | ALIGNED | `tests/test_heldout_pipeline.py` |
| P24 | Invariant 9 — tolerances are numbers set before a test runs: 3 Monte Carlo SEs; rates over 200 repetitions with a binomial 99% interval | ALIGNED | `controls.py:362` `SE_MULTIPLE = 3.0`, `:352` `REPETITIONS = 200`, `:353-354` `ALPHA`/`DETECTION_RATE` |

### Tasks

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| P25 | Task 1 — `masked_rowwise_pearson` moves to `fmharness.statistics`, re-exported so rung 0 is unchanged | ALIGNED | `statistics.py:97`; `delta_reproducibility.py:30` imports it and its other call sites are unchanged |
| P26 | Task 1 — `tahoe.py` ported with `parse_dose_um` and `scatter_tokens`, with tests | ALIGNED | `src/fmharness/tahoe.py` and `tests/test_tahoe.py` both present in the diff |
| P27 | Task 1 — dependencies `scikit-learn`, `scipy`, `rdkit` added; marker `step_fit` registered | ALIGNED | `pyproject.toml` diff adds all three and `step_fit: fitting a model on training data` |
| P28 | Task 1 — `MODELS` and `COMPARISONS` declared, with a test that LOLO has 6 comparisons and LODO 5 and that every comparison names declared models | ALIGNED | `__init__.py:53-92`; `tests/test_heldout_declarations.py:13,20` |
| P29 | Task 2 — `tests/fixtures/tahoe_drug_metadata.csv` holds the drug table's three columns for all 379 drugs at the pinned revision, as an input fixture | ALIGNED | 379 data rows (recount); columns `drug, pubchem_cid, canonical_smiles`; `DATA.md` records the revision `2dc5790…6546a95` |
| P30 | Task 2 — grid tests on the real promoted tables: 107 drugs, 50 lines, 5,350 pairs; `NA` is a line; `"Selinexor "` maps to `"Selinexor"`; exclusions are exactly the two A549 pairs; the four ceiling values | ALIGNED | every one independently recomputed here, including `NA in lines: True` and `"Selinexor " in grid drugs: True` |
| P31 | Task 2 — `heldout_grid.py` writes `rung1_grid.json` and `rung1_ceiling.csv` | ALIGNED | `scripts/heldout_grid.py` present with those outputs; artifacts are PR-1 |
| P32 | Task 3 — the answers scan reuses rung 0's DuckDB helpers, with `--part/--n-parts` slices and `--combine` | ALIGNED | `heldout_answers.py:80,111` call `dr._connect`; `:241` writes `answers_{part}.parquet`; `:228-233` the combine path |
| P33 | Task 3 — `--crosswalk` writes `rung1_line_crosswalk.csv`, covering the line whose DepMap key is the literal `NA` | ALIGNED | `heldout_answers.py` provides `--crosswalk`; the `NA` line survives the grid recompute (`keep_default_na=False`, `grid.py:221`) |
| P34 | Task 4 — cell keys, superset, plate-balanced selection, halves, pseudobulk | ALIGNED | `cells.py:58-178`, one function per contract element |
| P35 | Task 4 — the tranche's entry is added to `docs/DATA.md` (source, selection rule, script, date) in the change that produces it | ALIGNED | `DATA.md` diff adds "Tahoe-100M DMSO cells (rung 1)" with the source, the four-step selection rule, the scripts and the date |
| P36 | Task 4 — the cells are registered as tranche `tahoe100m-dmso-cells.v1` with `scripts/register_tranche.py` | **PENDING-RUN** | `scripts/register_tranche.py` exists, but `data/tranches/` holds only `tahoe100m-pseudobulk-de.v1`; `DATA.md` itself says "once a run registers it". Needs: the tranche JSON and manifest, with the content hash over the 50 per-line files |
| P37 | Task 5 — descriptions and chemistry interfaces, with the build control | ALIGNED | `descriptions.py`, `chemistry.py`; `test_build_identity_recovered`, `test_build_shuffled_identity_is_chance` |
| P38 | Task 6 — the embed script loads each checkpoint once, asserts one embedding per cell, and version 2 writes `rung1_weights_check.json`, failing if the encoders are identical | ALIGNED | `heldout_embed.py:378` and the `encoders_differ` contract; artifact is PR-4 |
| P39 | Task 7 — `LAMBDAS = np.logspace(-3, 3, 13)`, `Fit`, `drug_average`, `ridge_lolo`, `ridge_lolo_k`, `nearest_lines_lolo` | ALIGNED | `models.py:72,79-98,192,352,395,460` — every named interface present with the stated signature |
| P40 | Task 8 — `ridge_lodo` with kernel `T[d,d'] · (1 + K_line[l,l'])`; chemistry only is `ridge_lodo(zeros, T, ...)`; `ridge_lodo_k` | ALIGNED | `models.py:734,764`; `heldout_fit.py:367-369` builds chemistry only as the zero-kernel case, exactly as written |
| P41 | Task 9 — scoring and comparison interfaces, and the synthetic builders in `controls.py` | ALIGNED | `scoring.py:73,140`; `comparisons.py:98,119,172,235`; `controls.py:97-298` |
| P42 | Task 10 — a staged run on a 6-line × 8-drug fixture equals an in-process run, and a round whose outputs exist skips | ALIGNED | `test_staged_run_equals_one_process`; `records.py:48` `is_done` |
| P43 | Task 10 — combine writes every contract table, the control tables at the real grid's size and ceiling values, the figures, validated leakage records, and `rung1_run.params.json` | ALIGNED | `heldout_combine.py:991-995` names them; `:1181-1185` the five control tables; `:1196-1200` the five figures; `controls.py:325-333` uses the design's own R values and the screen's size. Artifacts are PR-8/PR-9/PR-10/PR-11/PR-12 |
| P44 | Task 11 — env: stack env on PATH, `PYTHONPATH`, HF cache and token, `RUNG1_CACHE`, `OUT_DIR`, and the array sizes in one place; CPU `acpu`/`cpu-normal` with memory in whole cores; GPU `ah200` | ALIGNED | `rung1_env.sh:29-30,43-45`; partitions and whole-core memory verified at D116/D119/D121 |
| P45 | Task 11 — a test checks each array's range against the env constants, that every job sources the env, and that every script is submitted by the chain with `--dependency=afterok` | **DEVIATION-RECORDED** | the fit and redraw arrays are pinned to the run's own module constants instead of the env file — `rung1_fit.sbatch` against `heldout_fit.py`'s `FULL_GRID_LINES`/`FULL_GRID_DRUGS`, `rung1_redraws.sbatch` against `comparisons.N_BLOCKS` (`tests/test_rung1_jobs.py`). Recorded in `decisions.md` (2026-09-11, ruling 43) with the reason; the data stages still read their sizes from the env file |
| P46 | Task 11 — a local synthetic dry run of the CLIs in order passes; push, `ralpine switch`, `ralpine update`, `ralpine ls`/`du` the inputs | ALIGNED | `verification.md:58-73` records the fixture run and the suite; `decisions.md` records `switch` having run on Alpine at 751f386/d62c76e with the moved-aside folders named |
| P47 | Task 11 — check the queues, submit the chain, confirm each dependency with `ralpine jobinfo`, monitor, resubmit failed indexes alone | **PENDING-RUN** | one job was submitted and died on the unmounted scratch after 113 s (`verification.md:31`); no chain has run. Needs: the submitted job ids, the `jobinfo` dependency read-backs, and the completed array |
| P48 | Task 11 — pull `OUT_DIR` tables and figures and every log; record jobs, nodes, wall times and memory in `verification.md` | **PENDING-RUN** | `verification.md:29-40` is the plan the run will be filled in from; every Status cell reads "not yet run". Needs: the pulled artifacts and the completed job table |
| P49 | Task 12 — the battery recomputes from the artifacts alone, prints claim / recomputed / pass-fail, and exits non-zero on any fail | ALIGNED | `verify_rung1.py:1537-1549` renders exactly that; `:1514-1534` `exit_status` returns 1 on any failure, on an all-skipped run, and on a skipped redraw family on the design's grid. Its pass over the real run is PR-15 |
| P50 | Task 12 — a test runs the battery on the Task 10 fixture outputs and skips on a clean checkout without run outputs | ALIGNED | `tests/test_verify_rung1.py`; `verify_rung1.py:1569-1572` returns 2 with a message when no run is present |
| P51 | Task 12 — `verify.ipynb` recomputes the same claims inline using standard library and pandas only, with no project imports, committed without outputs, its final cell calling the script | ALIGNED | its only imports are `csv, hashlib, json, math, os, statistics, subprocess, sys, pathlib, pandas`; the single `fmharness` occurrence is the string "fmharness (3.13.5)" in a version banner, not an import; 0 outputs and 0 execution counts |
| P52 | Task 12 — `verification.md` records the commands, outputs, job table and pointers to every artifact | ALIGNED | `verification.md:58-131`, with the job table at `:29-40` and the artifact pointers at `:112-130` |
| P53 | Task 13 — `review.md` records the code review's findings and dispositions | ALIGNED | not yet in the tree; the whole-branch review is dispatched separately and runs in parallel with this audit (brief, "Division of labour"). Accounted, not missing |
| P54 | Task 13 — the audit follows `docs/audit.md` and records `audit_checksums.json` for every artifact read | ALIGNED | this document and [`audit_checksums.json`](audit_checksums.json), 68 artifacts |
| P55 | Task 13 — `summary.ipynb` is committed without outputs | ALIGNED | 0 outputs, 0 non-null execution counts across its 10 code cells |
| P56 | Commits during execution: one commit per task on the working branch, no run output staged, the topic rebuild after gate 2 | ALIGNED | 5 commits on the branch beyond `dff524f`; no run output in the diff; the rebuild is deferred (D124) |

## Reverse-direction findings

All **69** changed files were accounted to a claim. Accounting in brief: the 10 documents to D131,
S1–S10 and P52–P55; `pyproject.toml` and `uv.lock` to P27; the 15 `src/fmharness` modules to
P25–P26 and P34–P43; the 11 rung 1 scripts to P31–P43 and P49; the 13 Alpine files to P44–P45 and
`decisions.md`'s "Tooling changed by this task" entry (for `ralpine`); the 16 test files to the
invariants and tasks that name them (`tests/test_ralpine_boundary.py` to the same tooling entry).

**Three files are under no claim.** None is a defect in itself; each is work the documents do not
describe.

| # | File | Finding |
|---|---|---|
| R1 | `scripts/heldout_fetch_metadata.py` (+75) | Named in **no** document — not `design.md`, `plan.md`, `decisions.md` or `verification.md` (grep count 0 in each). It downloads the pinned drug and gene metadata tables once so 64 DMSO tasks do not each open their own Hugging Face handle — a real and sensible stage, and it writes a `.done.json` like every other. It is absent from the plan's file map and from Task 11's file list, and no job in `submit_rung1_chain.sh` calls it, so how it is meant to be run is undocumented |
| R2 | `scripts/heldout_smoke_import.py` (+77) | Mentioned once, in `verification.md:72`, as a command that was run; named in no plan task and in no file map. Its own docstring calls it "the check task 11's jobs run first, before any long stage", which is a claim `plan.md`'s Task 11 does not make |
| R3 | `src/fmharness/heldout/records.py` (+63) | Absent from the plan's file map, though the "Data contracts" section describes the `.done.json` records it implements and `tests/test_heldout_records.py` covers it. The lightest of the three: the behaviour is documented, only the module is unlisted |

**Observation O1 (not a verdict).** The entry recording that the answers are scanned rather than
read from rung 0's cache is filed in `decisions.md` under the `## plan.md` heading, but it amends
`design.md` §9's stage 2 as well — that is the document whose text it contradicts. Project rule 2
asks that entries be "labeled by the document they amend". D109 is verdicted
DEVIATION-RECORDED because the entry exists, is dated and gives the reason; where it is filed is a
question for the fix wave, not a reason to downgrade it.

## The two rulings the controller asked for

### Ruling 1 — a notebook generated by a script that is not in the repository

**Finding.** `make_verify_nb.py` and `make_summary_nb.py` are absent from the tree and from every
diff on this branch (`git log --all --diff-filter=A` finds neither; `find` finds neither), yet they
surface in editor diagnostics, so they exist on the author's disk. They generate `verify.ipynb` and
`summary.ipynb`, which are committed.

**Ruling: this does not satisfy the lab's reproducibility standard, on a narrow ground.** The
committed notebook being the artifact of record is ordinary practice and is not the problem. The
problem is that a build step exists and only one machine has it. Two consequences follow, and both
are the failure the version-control rule exists to prevent:

1. **The notebook cannot be regenerated by anyone else.** The next person to change it either edits
   the generated file by hand — after which the generator and the notebook diverge silently, and
   the generator's next run destroys the hand edit — or re-derives the generator from scratch.
2. **The automation is not reviewable.** Whatever logic decides which claims the verification
   notebook recomputes lives outside the branch, so neither the code review nor this audit can read
   it. An audit can check that the notebook recomputes what it claims; it cannot check that the
   thing which wrote it will keep doing so.

**What limits the damage, and why it is still non-conforming.** The evidence does not depend on the
generators: both notebooks are committed without outputs (verified: 0 outputs, 0 execution counts)
and `verify.ipynb` recomputes every claim inline from the standard library and pandas with no
project imports (verified), so a reader executes it and sees their own numbers. Nothing scientific
is unreproducible. What is unreproducible is the *document*, and the lab standard — "all analyses
require reproducible automation" and "automation must be version controlled" — binds the automation
whether or not its output happens to be self-contained.

**Recorded, not fixed** (the controller dispatches the fix wave). The cheap disposition is to commit
the two generators under `scripts/`; the equally acceptable one is to delete them and declare the
notebooks hand-maintained source, since nothing in `design.md` or `plan.md` claims they are
generated. What is not acceptable is the current state, in which the tree is silent about a build
step that exists. Note this is *not* drift against any claim in scope — no document asserts the
generators exist — which is precisely why it needed a ruling rather than a table row.

### Ruling 2 — the exact semantics of `is_design_grid`

**The code** (`scripts/verify_rung1.py:266-274`, with `DESIGN_LINES, DESIGN_DRUGS = 50, 107` at
`:80`):

```python
def is_design_grid(record: dict[str, Any]) -> bool:
    """True unless this grid is plainly a smaller fixture than the design's. ..."""
    lines, drugs = len(record.get("lines", [])), len(record.get("drugs", []))
    return not (lines < DESIGN_LINES and drugs < DESIGN_DRUGS)
```

**Exact semantics, recorded so no later reader has to infer them.** The predicate is
`not (lines < 50 and drugs < 107)` — by De Morgan, `lines >= 50 or drugs >= 107`. It is **true
whenever the grid reaches the design's size in *either* dimension**, and false only when the grid is
smaller than the design in **both**. It is **not** an equality test, and its name reads like one.
Concretely: 50 × 8 is "design grid"; 3 × 107 is "design grid"; 60 × 200 is "design grid"; only a
grid under 50 lines *and* under 107 drugs is exempt. The docstring's "plainly a smaller fixture"
is accurate for the exempt case but does not state the disjunction.

**What keys off it** — five checks across four call sites, plus the exit status. The brief's "three
FAIL paths" is an undercount if checks are counted rather than call sites:

| Site | Effect when `is_design_grid` is true | Effect when false |
|---|---|---|
| `:340-356` | a strict check that `(lines, drugs) == (50, 107)` — **FAIL** if not | skipped as a fixture |
| `:358-392` (loop, **two** checks) | a missing `sha256_lines` / `sha256_drugs` is a **FAIL** | skipped |
| `:394` → `:398-414` | a missing `source_sha256` is a **FAIL** | skipped |
| `:1428-1464` | a missing or wrong `component_ks` is a **FAIL** | skipped, "fewer candidates are legitimate" |
| `:1514-1534`, `:1577` | any skipped check in the redraw family forces **exit 1** | that skip is tolerated |

**Ruling: correct as built, and it cannot misfire operationally — but the name is a hazard.** The
real run's grid is exactly 50 × 107 (recomputed here from rung 0's promoted table, with the one
46-line drug dropped), and every fixture in the tree is small in both dimensions (the 6 × 8 pipeline
fixture), so no input in existence lands in the asymmetric region. The asymmetry is also
*deliberate* and right: the comment at `:269-271` says "a run that lost a drug must FAIL here, not
skip", which is exactly what a 50 × 106 grid should do — it takes the strict path and then fails the
equality check at `:346`. A predicate that meant "exactly 50 × 107" would skip that run instead,
which is the bug this design avoids.

So: **not drift** (no document in scope asserts the helper's semantics), **not a behavioural
defect**, and recorded here because a later reader who assumes the name means equality would
misread five checks. If the fix wave touches it at all, the minimal change is to the *name and
docstring* — something like `is_at_least_design_scale`, with the disjunction spelled out — and not
to the logic, which should stay as it is.

## PENDING-RUN: the artifact checklist

The run has produced nothing. This is the checklist the audit delta works from once Alpine's storage
returns and the chain completes — not a re-enumeration. Each entry names the artifact, the claims it
will confirm, and the evidence the delta must record.

| # | Artifact | Confirms | Evidence the delta needs |
|---|---|---|---|
| PR-1 | `rung1_grid.json`, `rung1_ceiling.csv` | D11, D55–D58, P31 | the written ceiling rows equal 0.5815/0.7353/0.8575 and 0.0812/0.1503/0.3876; `sha256_lines`/`sha256_drugs`/`source_sha256` present and matching |
| PR-2 | `answers.npz`, `rung1_answer_counts.csv` | D7, D61 | every grid pair has `n_plates >= 2`; exactly 50 pairs have `n_plates == 3` (this audit could show only that 50 have an odd count) |
| PR-3 | `rung1_cells.csv`, `cells/line_*.h5ad` | D28, D29 | ≤ 1,000 cells per line; per-plate counts at or under the quota |
| PR-4 | `embedding_*.parquet`, `rung1_weights_check.json` | D24–D26, D88, P38 | the encoder checksums differ from `bc_large.ckpt`; **and the checkpoint filename actually loaded**, which settles D26 |
| PR-5 | `rung1_identity_match*.csv`, `rung1_identity_grid_*.csv` | D86, D87 | match above the shuffled 99th percentile; shuffled at about 1/50 |
| PR-6 | `scores_*.parquet`, `settings_*.csv` + `.done.json` | D42, P14 | 157 rounds present with valid records; chosen k drawn from {2,5,10,15,20} |
| PR-7 | `redraws_*.parquet` | D70, D71, D73, D75 | 8 blocks × 250 = 2,000 draws per contrast per gene set |
| PR-8 | `rung1_pair_scores.csv.gz`, `rung1_model_summary.csv`, `rung1_comparisons.csv`, `rung1_settings.csv` | D49, D63–D69, D74, D76–D78, D80 | the 11 declared comparisons plus the stand-in contrasts on both gene sets; `p_holm` filled only for declared × responding; the two excluded pairs absent |
| PR-9 | the five `rung1_control_*.csv` | D83, D89–D101 | each control's planted and recovered values, at the real grid's size and the design's R |
| PR-10 | `figures/01_build.png` … `05_null.png` | D89, D92, D95, D98, D101 | five non-empty PNGs, each drawn from the table beside it |
| PR-11 | `rung1_leakage_profiles.json` | D79–D82 | three profiles validating against `LeakageProfile`; `n_exposed_pairs == 2` |
| PR-12 | `rung1_run.params.json` | D42, P43 | producing commit, job id, seeds (incl. redraw base 7000), `component_ks`, input sha256s |
| PR-13 | `results/rung1-held-out-prediction/logs/` | **D107** | the pulled logs for every stage |
| PR-14 | `verification.md`'s job table | **P47, P48** | job ids, nodes, wall times, peak memory in place of "not yet run" |
| PR-15 | the battery against the real run | P49 | `verify_rung1.py` exiting 0 on the design's grid with no redraw-family skip |
| PR-16 | `data/tranches/tahoe100m-dmso-cells.v1.*` | **P36** | the tranche JSON and manifest, content hash over the 50 per-line files |

## Fix wave

**Dated** 2026-09-11. Four commits on top of the audit's `89bb680`: **b512bc2** (measurement),
**b4e6c39** (battery), **7cc3dc6** (the notebook generators, ruling 44), **73806e1** (documents).
HEAD is `73806e1`. Dispositions below; the re-audit that follows re-checks only the seven DRIFT
items, per the cap.

| Item | Disposition | Commit |
|---|---|---|
| D26 | **Fixed in the record, toward the file that is actually loaded.** `leakage.py`'s `STACK_CHECKPOINTS` and design §4 both now say `finetuned-epoch=5-val_loss=6.1078.ckpt`, matching `heldout_embed.py` and `CKPT_DRUG`. A dated `decisions.md` entry records it, on the ground that a provenance record naming a file that is not on the cluster pins nothing | `73806e1` |
| D108 | **Fixed in the document.** Design §9's stage table gains "0. Grid, ceiling and line crosswalk", and stages 1 and 2 now show they wait for it | `73806e1` |
| D113 | **Fixed in the document, deliberately not in the chain.** Stage 5's "Waits for" now reads "nothing, unless the operator passes `--after <job id>`". A dated entry gives the reason the chain stays as it is: the two stages are submitted days apart, so at data-submission time there is no fit job to attach to | `73806e1` |
| D131 | **Fixed per ruling 45** — reconciled to design §1–§7 (what gate 1 approved), not to `decisions.md`'s originally proposed text. A dated entry records field by field how the applied text had differed and why the design won, including the one applied change that was kept (the `Tasks` line, which this audit had verdicted ALIGNED at S10) | `73806e1` |
| S6 | **Fixed per ruling 46, in the sentence rather than the experiment.** Adding the missing comparisons would enlarge the Holm families of an approved design; the SPEC sentence now states design §7's actual structure. A dated entry records that neither the old SPEC text nor the proposed text matched §7 | `73806e1` |
| P17 | **Fixed by naming tests that exist, one of them new.** Invariant 2 now names `test_scoreable_boundary_49_versus_50_responding_genes`, `test_excluded_pairs_are_false_even_when_otherwise_scoreable`, and the newly written run-wide `test_the_excluded_pair_is_scored_for_no_model` | `73806e1` (plan.md), `b512bc2`/`73806e1` (the test) |
| P18 | **Fixed by writing the test the plan already named.** `test_descriptions_read_only_dmso_cells` now exists | `b512bc2` |

Beyond the seven, the wave also took up findings this audit raised outside the claim tables:

| Item | Disposition | Commit |
|---|---|---|
| R1, R2, R3 | **Fixed.** All three files under no claim entered plan.md's file map: `heldout_fetch_metadata.py`, `heldout_smoke_import.py` and `heldout/records.py`, each with the responsibility it carries | `73806e1` |
| O1 | **Fixed.** The "answers are scanned, not cached" entry moved from `decisions.md`'s `## plan.md` heading to `## design.md (execution)`, the document whose text it contradicts, with a line saying why it moved. Nothing about the decision changed | `73806e1` |
| Audit ruling 1 (ruling 44) | **Fixed.** Both generators committed to `scripts/`, entered in the file map, recorded in `decisions.md`, and pinned by `tests/test_notebook_generators.py` | `7cc3dc6` |
| Audit ruling 2 (ruling 47) | **Fixed.** `is_design_grid` renamed `is_at_least_design_size`; logic unchanged | `b4e6c39` |

## Re-audit

**Date** 2026-09-11. **Commit re-audited** `73806e1`. **Auditor** the same fresh reader.
**Scope** the **seven items verdicted DRIFT** and nothing else. Per `docs/audit.md`'s cap this pass
does not re-enumerate the design and does not revisit anything verdicted ALIGNED,
DEVIATION-RECORDED or PENDING-RUN; there is no confirmation pass of a confirmation pass.

**What was re-run.** The named tests behind the two invariant items, the battery and job tests, the
notebook-generator tests, and project rule 2 (plan.md's invariants were rewritten, which is a
non-additive task-document edit):

```
$ uv run pytest tests/test_notebook_generators.py tests/test_rung1_jobs.py \
                tests/test_verify_rung1.py -W error        108 passed, 9 skipped
$ uv run pytest tests/test_heldout_cells.py::test_descriptions_read_only_dmso_cells \
      tests/test_heldout_answers.py::test_scoreable_boundary_49_versus_50_responding_genes \
      tests/test_heldout_answers.py::test_excluded_pairs_are_false_even_when_otherwise_scoreable
                                                            3 passed
$ uv run pytest tests/test_heldout_pipeline.py::test_the_excluded_pair_is_scored_for_no_model
                                                            1 passed (84 s)
$ uv run pytest tests/test_project_rules.py -k rule_02      2 passed, 7 deselected
```

<sub>A first attempt at the rule‑2 run was wrapped in `timeout`, which does not exist on macOS; the
command failed without running pytest and the shell reported the exit status of `tail`. It was
re-run unwrapped, and the output above is from that run.</sub>

| Item | Re-audit verdict | Evidence |
|---|---|---|
| D26 | **ALIGNED** | All five places that name the drug checkpoint now agree on `finetuned-epoch=5-val_loss=6.1078.ckpt`: `leakage.py:45`, `design.md:53`, `heldout_embed.py:41`, `rung1_env.sh:39` (`CKPT_DRUG`), and `test_heldout_pipeline.py:351`, which pins it. Recorded in `decisions.md` under `## design.md (execution)`. The file the cluster actually opens is confirmed only by the run (checklist PR-4), and the entry says so |
| D108 | **ALIGNED** | `design.md:195` now carries "0. Grid, ceiling and line crosswalk \| one job; both data stages read the `rung1_grid.json` it writes \| —", and `:196-197` show stages 1 and 2 waiting for 0 — matching `submit_rung1_chain.sh:140,147,160`, where both data stages take `--dependency=afterok:$GRID` |
| D113 | **ALIGNED** | `design.md:200` now reads "nothing, unless the operator passes `--after <job id>` (`decisions.md`, 2026-09-11)", which is what `submit_rung1_chain.sh:25-26` and `:98-100` implement. The document no longer claims a dependency the chain does not create, and the dated entry explains why the chain stays that way |
| D131 | **ALIGNED** | `git diff 89bb680..HEAD -- docs/SPEC.md` shows *Adds*, *Measure* and *Why it matters* restored to `decisions.md`'s proposed wording (including the label *Why it matters*, which the applied text had renamed), *Tasks* kept, and *Reports* rewritten under ruling 46. The dated entry walks all six fields and says which way each went and why |
| S6 | **ALIGNED** | The SPEC sentence now describes design §7's actual structure, and I checked it clause by clause against `__init__.py:81-91` and `heldout_redraws.py:113-126`: Stack base against the drug average (LOLO) and chemistry only (LODO); Stack base against expression, PCA, NMF, and nearest lines under LOLO only; drug fine-tune against the cytokine release; and each description against its own stand-in. Every clause holds, and no comparison is promised that `contrasts()` does not compute. See the note below on one loose word |
| P17 | **ALIGNED** | Invariant 2 (`plan.md:92-95`) names three tests, all of which exist and pass: the two boundary tests in `tests/test_heldout_answers.py` and the new run-wide `tests/test_heldout_pipeline.py:754`, which asserts the excluded pair appears in no model's scores in either scheme on either gene set, with `assert len(scores) > 0` guarding against a vacuous pass. `test_scoreable_pairs_do_not_depend_on_the_model` no longer appears in any document but this audit's own record of the original finding |
| P18 | **ALIGNED** | `tests/test_heldout_cells.py:603` now defines `test_descriptions_read_only_dmso_cells`, and it is not vacuous: `:662` asserts the line's description equals the pseudobulk of its DMSO cells alone (`rtol=1e-12`), and `:663-666` then asserts that folding the treated cell in **would** have changed it — so the test cannot pass by measuring nothing. The fixture deliberately carries a treated cell and a non-grid line, and `:645` checks the non-grid line was not described |

**Verifying ruling 44's asserted part.** The controller flagged that the generators were
*reconstructed*, so "they regenerate the notebooks byte-for-byte" was a claim that could be asserted
rather than true. I checked it independently rather than trusting `tests/test_notebook_generators.py`
— I ran both generators to a scratch directory and compared bytes:

```
verify.ipynb   committed=49752 bytes sha=7ed68fbea291727f
               generated=49752 bytes sha=7ed68fbea291727f    IDENTICAL
summary.ipynb  committed=37249 bytes sha=a01073e5c6af4497
               generated=37249 bytes sha=a01073e5c6af4497    IDENTICAL
```

Both are tracked (`git ls-files scripts/make_verify_nb.py scripts/make_summary_nb.py`), both are in
plan.md's file map, and `decisions.md` states plainly that they were reconstructed rather than
moved. The test itself compares bytes rather than parsed cells, and separately pins that the
generated notebooks carry no outputs and that `--out` works. **The claim is true**, and the
reproducibility gap my first ruling identified is closed: anyone can now regenerate either notebook,
and a hand-edit that bypasses the generator fails the suite.

**Note on S6's one loose word.** The new sentence groups "nearest lines" among "the other line
descriptions", then says "every description also goes against a random stand-in". Nearest lines has
no stand-in — `stand_in_contrasts` filters on `spec.kind == "ridge"` and `MODELS` has no
`random_nearest_lines`. Read in the design's own vocabulary the sentence is correct, because §4's
table defines the six *descriptions* and §5 lists nearest lines as a *model*, so "every description"
denotes the six, each of which does face a stand-in. I record the ambiguity rather than reopen the
item: the defect S6 named — a comparison promised that the run does not compute — is gone, and this
is a wording preference for a later documentation pass, not a claim the tree contradicts.

**Note on ruling 47.** `is_at_least_design_size` (`verify_rung1.py:266-280`) keeps the predicate
exactly (`not (lines < DESIGN_LINES and drugs < DESIGN_DRUGS)`) and the docstring now states the De
Morgan form, the deliberate asymmetry, and that five checks across four call sites and the exit
status depend on it. Every call site moved with it; `is_design_grid` survives nowhere in `scripts/`,
`src/`, `tests/` or `docs/` except this audit's own record of the original finding, which is
historical and correct as written. My Ruling 2 above suggested the name `is_at_least_design_scale`;
the implementer chose `is_at_least_design_size`, which is the same thing.

**On `.superpowers/sdd/plan/global-constraints.md:85` — I accept the controller's ruling, with one
caveat recorded.** The facts check out: `.superpowers/sdd/.gitignore` is a single `*`,
`git check-ignore -v` confirms the file is ignored by that line, and `git ls-files .superpowers/`
returns nothing, so no part of that tree is tracked. The authoritative statement of invariant 6 is
`plan.md:100-105`, which is tracked, correct, and has been since `7ce7660`. A reader of a fresh
clone therefore never encounters the wrong statement — which is the outcome my audit exists to
protect, so the correction being absent from git costs that reader nothing.

The caveat is the opposite risk, and it is why I record this rather than simply agreeing. The file
is not inert: `plan.md:3` directs agentic workers to the superpowers workflow, and the task brief
told this task's workers to read `global-constraints.md` as binding. A binding constraint document
that is untracked cannot be reviewed, cannot be diffed, and does not carry its correction forward if
the workspace is ever recreated — so the same reversed statement can return silently and no test
will catch it. That is a workspace-tooling concern rather than a rung 1 defect, and it is out of
this audit's scope; I note it so a future reader knows the correction was made on disk on
2026-09-11 and is not in the history.

### Verdict

**Every one of the seven drift items is fixed. The audit is PASSED.**

The clause tables above are left as they were written — they are the record of what the audit found
at `55adfb7`, and editing them would erase the findings. For those seven claims the verdicts in this
section supersede them, so the standing tally at `73806e1` is:

| Verdict | Count |
|---|---|
| ALIGNED | 187 |
| DEVIATION-RECORDED | 6 |
| DRIFT | **0** |
| PENDING-RUN | **4** |
| **Total** | **197** |

**What is still open, and is not a defect.** The run has not happened: D107, P36, P47 and P48 remain
PENDING-RUN, and the 55 claims verdicted on their code still await the artifact that realizes them.
Nothing may be promoted on the strength of this pass. When Alpine's storage returns and the chain
completes, that work gets an **audit delta** under its own dated heading in this file — the same
procedure over the new surface only, with the same fresh-reader confirmation, working from the
16-entry [PENDING-RUN artifact checklist](#pending-run-the-artifact-checklist) rather than
re-enumerating the design.
