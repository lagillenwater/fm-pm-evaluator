# Rung 1 — implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development, one task at a time, with
> superpowers:test-driven-development inside each task. Steps use checkbox (`- [ ]`) syntax.

**Goal** Build and run rung 1 as `design.md` specifies — hide one line (50 rounds) or one drug (107 rounds) at
5 uM, fit every model, score per pair, compare with redraws — and stop at `summary.ipynb` for gate 2.
**Architecture** Pure, tested numerical code in `src/fmharness/heldout/`; thin command-line scripts in
`scripts/heldout_*.py` that read and write files; Alpine job scripts in `scripts/alpine/rung1_*` that run
those scripts as job arrays. Rung 0's code is reused, never changed in behavior.
**Tech stack** Python 3.11, numpy, pandas, scipy, scikit-learn (NMF), RDKit (fingerprints), DuckDB and
pyarrow (the answers scan), anndata and Stack (`arc_stack` 0.1.3, Alpine only), matplotlib.
**Spec** [`design.md`](design.md) (approved 2026-09-11). **Lineage** [`decisions.md`](decisions.md).
**As of** 2026-09-11.

## Global constraints (copied from the design; every task inherits them)

- Dose **5.0 uM** only. Grid **107 drugs × 50 lines = 5,350 pairs**: drugs with a replicated 5 uM triple in all
  50 lines of rung 0's promoted `results/rung0-assay-reliability/rung0_per_pair_r.csv`.
- Answer = mean `log2FoldChange` over the pair's plates. Responding genes: `padj < 0.05` on at least one plate.
  All genes: every gene with a finite answer. A pair is scored on a gene set when **≥ 50** genes qualify.
- Line descriptions: expression (log2(CPM+1) over Stack's 15,012 genes, standardized across 50 lines), PCA,
  NMF (k ∈ {2, 5, 10, 15, 20}), Stack base / cytokine / drug (mean embedding, standardized), a random
  stand-in of the same width per description. Up to 1,000 DMSO cells per line, plate-balanced, fixed seed.
- Models: drug average (LOLO), chemistry only (LODO), ridge for every description (both schemes), nearest
  lines (LOLO, k ∈ {3, 5, 10, 20}). Settings chosen inside the round from training data only.
- Comparisons: design §7 table (6 LOLO, 5 LODO), responding genes, 2,000 redraws of the held-out unit, CI =
  2.5/97.5 percentiles, two-sided p, MDE = 2.8 × redraw SD, Holm within each test, two-way design effect.
- Leakage: pairs (ACH-000681, Trametinib) and (ACH-000681, Temsirolimus) are removed for every model.
- Ceiling √SB from the promoted per-pair table: responding 0.8575, all 0.3876 (design §6 table).
- Run outputs never enter git (until the evidence-storage change). Stage files by name; never `git add -A`.
- New code passes `ruff check`, `ruff format --check` and `pyright` (strict, `src` and `tests`).

## File map

| Path | Responsibility |
|---|---|
| `src/fmharness/statistics.py` | gains `masked_rowwise_pearson`, moved from `scripts/delta_reproducibility.py` (re-exported there, so rung 0 is unchanged) |
| `src/fmharness/tahoe.py` | ported from the archive: `parse_dose_um`, `scatter_tokens` |
| `src/fmharness/heldout/__init__.py` | package marker, the model and comparison declarations (`MODELS`, `COMPARISONS`) |
| `src/fmharness/heldout/grid.py` | the grid, drug-name crosswalk, leakage pairs, ceiling table, restriction record |
| `src/fmharness/heldout/answers.py` | answer matrices from the scanned rows; scoreable-pair masks |
| `src/fmharness/heldout/cells.py` | DMSO cell keys, plate-balanced selection, halves, pseudobulk log CPM |
| `src/fmharness/heldout/descriptions.py` | standardizing, PCA, NMF, random stand-ins, normalized linear kernel, identity match |
| `src/fmharness/heldout/chemistry.py` | Morgan fingerprints and Tanimoto similarity |
| `src/fmharness/heldout/models.py` | drug average, ridge (LOLO), nearest lines, ridge (LODO) with closed-form tuning |
| `src/fmharness/heldout/scoring.py` | per-pair scores for both gene sets; fraction of ceiling |
| `src/fmharness/heldout/comparisons.py` | redraw contrasts, Holm, two-way design effect |
| `src/fmharness/heldout/controls.py` | synthetic grids with planted answers, shared by tests and the combine job |
| `src/fmharness/heldout/figures.py` | every design §8 figure, each drawn from a written table |
| `src/fmharness/heldout/leakage.py` | `LeakageProfile` records per model version |
| `scripts/heldout_grid.py` | writes grid, ceiling and restriction record (runs locally or on Alpine) |
| `scripts/heldout_answers.py` | one gene slice of the answers scan (reuses rung 0's DuckDB helpers); `--combine` |
| `scripts/heldout_dmso_cells.py` | one shard block of DMSO cells from Hugging Face; `--combine` |
| `scripts/heldout_embed.py` | Stack embeddings per line and per half, one model load per version (Alpine only) |
| `scripts/strip_ckpt_head.py` | ported from the archive: encoder-only copy of an aligned checkpoint |
| `scripts/heldout_descriptions.py` | expression, PCA, NMF, random stand-ins, fingerprints, identity-match table |
| `scripts/heldout_fit.py` | one round (`--scheme lolo|lodo --round i`): every model, per-pair scores, chosen settings |
| `scripts/heldout_redraws.py` | one block of redraws for every comparison |
| `scripts/heldout_combine.py` | gathers rounds and redraws; writes result tables, control tables, figures, params sidecar |
| `scripts/verify_rung1.py` | the verification battery (claim / recomputed / pass-fail) |
| `scripts/alpine/rung1_env.sh`, `rung1_*.sbatch`, `submit_rung1_chain.sh` | the job chain of design §9 |
| `tests/test_heldout_*.py`, `tests/test_rung1_controls.py`, `tests/test_verify_rung1.py`, `tests/test_tahoe.py` | unit and known-answer tests |

## Data contracts between stages

Scratch root `RUNG1_CACHE=/scratch/alpine/$USER/rung1_cache`; result tables in `OUT_DIR=docs/tasks/rung1-held-out-prediction`
(untracked). Every file a stage writes is followed by a `<name>.done.json` completion record holding its sha256,
so a rerun skips finished work and the audit can pin bytes.

| File | Written by | Contents |
|---|---|---|
| `rung1_grid.json` | grid | `lines` (50 DepMap keys incl. literal `NA`), `drugs` (107 screen names, verbatim incl. trailing spaces), `metadata_name` (screen name → drug-table name), `excluded_pairs`, `sha256_lines`, `sha256_drugs`, `source_sha256` |
| `rung1_ceiling.csv` | grid | `gene_set, pairs_scored_rung0, split_half_r, sb, sqrt_sb, promoted_r, promoted_sb, promoted_sqrt_sb` |
| `answers_{k}.parquet` → `answers.npz` | answers | rows `line, drug, gene, mean_lfc, min_padj, n_lfc, n_plates`; combined arrays `lines, drugs, genes, delta[L,D,G] float32 (NaN = untested), responding[L,D,G] bool`; `rung1_answer_counts.csv` per pair |
| `dmso_{t}.parquet` + `dmso_{t}.npz` → `cells/line_{i}.h5ad`, `expression.parquet`, `expression_halves.parquet`, `rung1_cells.csv` | DMSO cells | selected cells' raw counts over the Stack panel, obs `cellosaurus, line, plate, key, half`; per-line and per-half log2(CPM+1); cells per line and plate |
| `embedding_{version}.parquet`, `embedding_{version}_halves.parquet`, `rung1_weights_check.json` | embed | per-line mean embedding (50 × width); per (line, half); drug-vs-base encoder checksum comparison |
| `descriptions.npz`, `tanimoto.npz`, `rung1_identity_match.csv` | descriptions | every description matrix and its random stand-in; 107 × 107 similarity in grid drug order; half-vs-half match for every description with its shuffled null |
| `scores_{scheme}_{i:03d}.parquet`, `settings_{scheme}_{i:03d}.csv` | fit | `scheme, round, model, line, drug, gene_set, r, n_genes`; `model, lambda, k, loss_min, lambda_at_edge` |
| `redraws_{b}.parquet` | redraws | `comparison, gene_set, draw, estimate` (one-way) and `estimate_two_way` |
| `rung1_pair_scores.csv.gz`, `rung1_model_summary.csv`, `rung1_comparisons.csv`, `rung1_settings.csv`, `rung1_control_{build,split,fit,score,null}.csv`, `rung1_leakage_profiles.json`, `rung1_run.params.json`, `figures/*.png` | combine | the result tables of design §9 |

## Invariants (each has a named test)

1. **No held-out answer reaches fitting or tuning.** Under LOLO nothing reads `delta[held]`; under LODO nothing
   reads `delta[:, held]`. Test: replace the held-out entries with arbitrary values; predictions are
   bit-identical (`test_lolo_ignores_held_out_answers`, `test_lodo_ignores_held_out_answers`).
2. **Every model is scored on the same pairs.** Scoreability depends on the answer alone; excluded pairs are
   removed before scoring (`test_scoreable_pairs_do_not_depend_on_the_model`).
3. **Descriptions carry no drug response.** Built only from DMSO cells (`test_descriptions_read_only_dmso_cells`).
4. **Determinism.** Cell selection, random stand-ins, NMF and redraws are seeded; reruns are bit-identical
   (`test_cell_selection_is_independent_of_shard_order`, `test_redraws_are_seeded`).
5. **Ceiling values reproduce the design table** from the promoted per-pair table (`test_ceiling_table_matches_design`).
6. **Tuning is closed-form and exact.** Ridge leave-one-line-out and leave-one-drug-out losses equal brute-force
   refits (`test_ridge_lolo_closed_form_loo_equals_refits`, `test_ridge_lodo_block_loo_equals_refits`).
   Tuning holds the round's reference (drug average, or line means) fixed at its training-set value.
7. **The linear kernel is scaled to a mean diagonal of 1**, so the LODO term `1 + k(l, l')` weighs the two parts
   comparably. Tuning losses are mean squared error over tested entries only.
8. **Staged equals one process.** Rounds + combine on a fixture equal a single in-process run
   (`test_staged_run_equals_one_process`), per PROCESS §2.
9. **Tolerances are numbers, set before a test runs.** A known-answer test passes when the recovered value is
   within 3 Monte Carlo standard errors of the planted value, the standard error estimated from the synthetic
   draw itself (for a mean of correlations: their SD over √n). Rates (80% detection, ≤ 5% false positives) use
   200 synthetic repetitions and a binomial 99% interval.

## Tasks

### Task 1 — Shared pieces and project settings

**Files** modify `src/fmharness/statistics.py`, `scripts/delta_reproducibility.py` (import only), `pyproject.toml`;
create `src/fmharness/tahoe.py`, `src/fmharness/heldout/__init__.py`, `tests/test_tahoe.py`.
**Interfaces** `masked_rowwise_pearson(a, b, min_genes, *, select=None) -> np.ndarray` (unchanged signature, new home).
`parse_dose_um(s: str) -> float`; `scatter_tokens(genes_list, expr_list, token_to_col, n_cols) -> csr_matrix`.
`MODELS: dict[str, ModelSpec]` and `COMPARISONS: tuple[Comparison, ...]` declaring design §5 and §7 verbatim
(ids: `drug_average, chemistry_only, expression, pca, nmf, nearest_lines, stack_base, stack_cytokine,
stack_drug`, and `random_<description>` for the six descriptions).

- [ ] Move `masked_rowwise_pearson` to `fmharness.statistics`; `delta_reproducibility` imports it. Run
  `uv run pytest tests/test_rung0_controls.py -q` — all pass, unchanged.
- [ ] Port `tahoe.py` with tests: dose string `"[('X',0.05,'uM')]"` → 0.05, malformed → NaN; tokens off the
  panel and the leading marker dropped; a two-cell fixture scatters to the expected dense matrix.
- [ ] Add dependencies `scikit-learn>=1.4`, `scipy>=1.11`, `rdkit>=2024.3`; register marker
  `step_fit: fitting a model on training data`. `uv sync --extra dev`.
- [ ] Declare `MODELS` and `COMPARISONS`; test that the LOLO family has 6 comparisons and LODO 5, and that every
  comparison names declared models.
- [ ] Gates: ruff, format, pyright over tracked files; full suite passes. Commit.

### Task 2 — The grid, the ceiling, the restriction record

**Files** `src/fmharness/heldout/grid.py`, `scripts/heldout_grid.py`, `tests/test_heldout_grid.py`.
**Interfaces** `Grid(lines: tuple[str,...], drugs: tuple[str,...], metadata_name: dict[str,str],
excluded_pairs: tuple[tuple[str,str],...])`; `grid_from_rung0(per_pair: DataFrame, dose=5.0) -> Grid` (without
names or exclusions); `attach_drug_metadata(grid, drug_metadata) -> Grid` (match on `str.strip()`, raise on any
unmatched); `sciplex_exposed_pairs(grid, drug_metadata, line="ACH-000681", cids=SCIPLEX_CIDS)`;
`ceiling_table(per_pair, grid, dose_strata) -> DataFrame`; `restriction_record(grid, sources) -> dict`.

- [ ] Add `tests/fixtures/tahoe_drug_metadata.csv`: the drug table's `drug, pubchem_cid, canonical_smiles`
  columns for all 379 drugs at Hugging Face revision `2dc57900b7981cfcf5e211527169a0b006546a95`, so tests run
  offline. It is an input fixture, not a run output.
- [ ] Tests on the real promoted tables (committed on this branch): 107 drugs, 50 lines, 5,350 pairs; `NA` is a
  line; `"Selinexor "` maps to `"Selinexor"`; exclusions are exactly the two A549 pairs; ceiling rows equal
  0.5815 / 0.7353 / 0.8575 (4,593 pairs) and 0.0812 / 0.1503 / 0.3876 (5,350), with the promoted 5 uM row beside.
  A synthetic per-pair table with one gap raises. Implement, pass.
- [ ] `scripts/heldout_grid.py --drug-metadata <parquet> --out-dir <dir>` writes `rung1_grid.json` and
  `rung1_ceiling.csv`. The drug table comes from Hugging Face revision `2dc57900b7981cfcf5e211527169a0b006546a95`.
- [ ] Gates; commit.

### Task 3 — The answers: scan and assemble

**Files** `src/fmharness/heldout/answers.py`, `scripts/heldout_answers.py`, `tests/test_heldout_answers.py`.
**Interfaces** `slice_answers(paths, drugs, lines, dose, tmp, *, n_parts, part, memory_limit, threads) -> DataFrame`
(script; reuses `_connect`, `_pool_columns`, `_drug_predicate`, `_gene_partition`, `_compact_df` from rung 0;
groups by (line, drug, gene) over rows with a plate at the dose; `mean_lfc = avg(log2FoldChange)`,
`min_padj = min(padj)`, `n_lfc = count(log2FoldChange)`, `n_plates = count(DISTINCT plate)`).
`Answers(lines, drugs, genes, delta, responding)`; `assemble_answers(rows, grid) -> Answers`;
`scoreable(answers, excluded_pairs, min_genes=50) -> dict[str, ndarray[L,D] bool]`.

- [ ] Fixture pool (as rung 0's `write_fixture_pool`): three doses, two plates, planted `log2FoldChange` and
  `padj`. Tests: mean over plates exact; one untestable plate → mean of the other; responding when either plate
  < 0.05; other doses and drugs excluded; the `NA` line kept; slices concatenate to the one-pass frame exactly
  (`test_answer_slices_equal_one_pass`); untested genes are NaN in `delta`; a pair with 49 responding genes is
  not scoreable for `responding` but is for `all`.
- [ ] CLI: `--part k --n-parts 8` writes `answers_{k}.parquet` with its per-task spill directory; `--combine`
  writes `answers.npz` and `rung1_answer_counts.csv`, refusing if any slice is missing.
- [ ] `--crosswalk` writes `rung1_line_crosswalk.csv` (`line, cellosaurus, cell_name`): the distinct
  `Cell_ID_DepMap, Cell_ID_Cellosaur, Cell_Name_Vevo` of the grid lines, read from the key columns alone. It is
  how Task 4 finds each line's cells, and it covers the line whose DepMap key is the literal `NA`. Test: one row
  per grid line on the fixture pool; a line with two Cellosaurus ids raises.
- [ ] Gates; commit.

### Task 4 — DMSO cells: selection, halves, pseudobulk

**Files** `src/fmharness/heldout/cells.py`, `scripts/heldout_dmso_cells.py`, `tests/test_heldout_cells.py`.
**Interfaces** `cell_keys(shard_index, row_group, rows, seed) -> ndarray[uint64]` (splitmix64 of a packed 64-bit
position); `superset_mask(keys, fraction=0.25) -> ndarray[bool]`; `select_cells(meta: DataFrame[line, plate, key],
per_line=1000) -> ndarray[bool]` (quota `ceil(per_line / plates_of_line)` smallest keys per (line, plate), then the
`per_line` smallest per line); `half_of(keys) -> ndarray[int8]` (bit 1 of the key); `pseudobulk_log_cpm(counts,
groups) -> (labels, ndarray)`.

- [ ] Tests: keys are identical whatever order shards are read in; quota and cap exact on a fixture with uneven
  plates (a plate with fewer cells than its quota gives all it has); halves are about equal and independent of
  selection; pseudobulk equals a hand-computed CPM on a 3-cell fixture.
- [ ] CLI block mode: `--block t --n-blocks 64` reads shards `t`-th of 64 (sorted `data/*.parquet`), keeps
  `drug == "DMSO_TF"` cells of the 50 grid lines (Cellosaurus ids from `rung1_line_crosswalk.csv`) whose key is in
  the superset, decodes only those row groups, and writes `dmso_{t}.parquet` (metadata) and `dmso_{t}.npz`
  (counts over the Stack panel); it also writes the count of all DMSO cells per (line, plate) seen.
  `--combine` selects, writes one `cells/line_{i}.h5ad` per line (obs `cellosaurus, line, plate, key, half`,
  var `feature_name`), `expression.parquet`, `expression_halves.parquet`, `rung1_cells.csv`, and fails if any
  (line, plate) superset holds fewer cells than its quota while the full count holds more.
- [ ] Register the cells as tranche `tahoe100m-dmso-cells.v1` with `scripts/register_tranche.py` (Hugging Face
  revision `2dc57900b7981cfcf5e211527169a0b006546a95`, content hash over the per-line files and the drug table),
  and add its entry to `docs/DATA.md` (source, selection rule, script, date) in the change that produces it.
- [ ] Gates; commit.

### Task 5 — Line descriptions and chemistry

**Files** `src/fmharness/heldout/descriptions.py`, `src/fmharness/heldout/chemistry.py`,
`scripts/heldout_descriptions.py`, `tests/test_heldout_descriptions.py`.
**Interfaces** `standardize(X) -> ndarray` (zero-variance columns become 0); `pca_components(X_std, k_max=20)`
(SVD, sign fixed so each component's largest loading is positive); `nmf_components(X_nonneg, k, seed)`;
`random_stand_in(n_lines, width, seed)`; `linear_kernel(Z) -> ndarray[L,L]` (scaled to mean diagonal 1);
`identity_match(half_a, half_b) -> float` (share of lines whose most correlated other-half profile is their own);
`shuffled_identity_null(cells_by_half, n_shuffles, seed) -> ndarray`; `morgan_fingerprints(smiles, radius=2,
n_bits=1024)`; `tanimoto(F) -> ndarray[D,D]`.

- [ ] Tests: PCA matches `numpy.linalg.svd` up to fixed sign; NMF reproducible under a seed; stand-in width and
  seed; kernel mean diagonal 1; Tanimoto of a molecule with itself 1, of two disjoint fingerprints 0, of a hand
  pair its hand value; unparseable SMILES raises.
- [ ] **build control (known answer):** synthetic lines with distinct profiles → identity match 1.0 and above the
  shuffled 99th percentile; shuffled labels → about 1/50 (`test_build_identity_recovered`,
  `test_build_shuffled_identity_is_chance`).
- [ ] CLI writes `descriptions.npz` (expression; PCA 20; NMF for each k; each random stand-in), `tanimoto.npz`
  in grid drug order, and `rung1_identity_match.csv` for expression, PCA, NMF (Stack rows are added by Task 6).
- [ ] Gates; commit.

### Task 6 — Stack embeddings (Alpine GPU)

**Files** `scripts/heldout_embed.py`, `scripts/strip_ckpt_head.py` (ported), `tests/test_heldout_embed.py`.
**Interfaces** `line_means(cell_embeddings, lines, halves) -> (per_line DataFrame, per_half DataFrame)`;
`encoders_differ(state_a, state_b, prefix_filter) -> dict` (per-tensor sha256 over shared encoder keys).
The script loads each checkpoint once with `stack.model_loading.load_model_from_checkpoint`, calls
`model.get_latent_representation` on each `cells/line_{i}.h5ad` (so every group holds one line), asserts one
embedding per cell, and writes `embedding_{version}.parquet` and `_halves.parquet`; version 2 (drug) also writes
`rung1_weights_check.json` against `bc_large.ckpt`, failing if the encoders are identical. It appends Stack rows
to `rung1_identity_match.csv` as `rung1_identity_match_stack_{version}.csv`.

- [ ] Local tests for the pure functions (means and halves on a fixture; identical and differing state dicts).
- [ ] Gates; commit. (The GPU run is Task 11.)

### Task 7 — Models when a line is hidden

**Files** `src/fmharness/heldout/models.py`, `tests/test_heldout_models.py`, `tests/test_rung1_controls.py`.
**Interfaces** `LAMBDAS = np.logspace(-3, 3, 13)`; `Fit(prediction: ndarray, lam: float|None, k: int|None,
loss_min: float, at_edge: bool)`; `drug_average(delta0, train) -> ndarray[D,G]`;
`ridge_lolo(K, delta0, tested, held, lambdas=LAMBDAS) -> Fit` (per-drug ridge on departures from the training
drug average, one λ shared across drugs and genes, exact leave-one-line-out loss via the hat-matrix diagonal);
`ridge_lolo_k(Zs: dict[int, ndarray], delta0, tested, held) -> Fit` (joint k and λ); `nearest_lines_lolo(S,
delta0, tested, held, ks=(3,5,10,20)) -> Fit`. `delta0` is `delta` with untested entries set to 0.

- [ ] Tests: ridge prediction equals `sklearn.linear_model.Ridge(fit_intercept=False)` on centred data in primal
  form; closed-form loss equals brute-force refits; invariant 1; nearest lines with k = all training lines equals
  the drug average; on a small grid with a strong planted line-linear response, ridge beats the drug average in
  squared error (the full known-answer control, placed relative to the MDE, is in Task 9).
- [ ] Gates; commit.

### Task 8 — Models when a drug is hidden

**Files** `src/fmharness/heldout/models.py`, `tests/test_heldout_models.py`, `tests/test_rung1_controls.py`.
**Interfaces** `ridge_lodo(K_line, T, delta0, tested, held, lambdas=LAMBDAS, line_term=True) -> Fit` with
kernel `T[d,d'] · (1 + K_line[l,l'])` on departures from each line's training-drug mean; chemistry only is
`ridge_lodo(zeros, T, ...)`; `ridge_lodo_k(Zs, T, ...) -> Fit`. The solution rotates the answers by the
eigenvectors of the two kernels; tuning uses the exact leave-one-drug-out residuals from the block of the hat
matrix, which shares the line eigenvectors.

- [ ] Tests: prediction equals a dense solve on a 6 × 8 × 5 grid; block leave-one-drug-out loss equals refits;
  invariant 1 for drugs; with `K_line = 0` and `T` = identity the prediction equals each line's training mean.
  (The known-answer fit control for drugs is in Task 9.)
- [ ] Gates; commit.

### Task 9 — Scoring and comparisons

**Files** `src/fmharness/heldout/scoring.py`, `src/fmharness/heldout/comparisons.py`,
`src/fmharness/heldout/controls.py`, `tests/test_heldout_scoring.py`, `tests/test_rung1_controls.py`.
**Interfaces** `score_pairs(pred[L,D,G], answers, masks) -> DataFrame[line, drug, gene_set, r, n_genes]`
(uses `masked_rowwise_pearson`, min 50 genes, excluded pairs dropped); `fraction_of_ceiling(mean_r, sqrt_sb)`;
`redraw_estimates(diffs, unit_index, n_units, n_draws, seed) -> ndarray` (multinomial unit weights);
`summarize_redraws(estimate, draws) -> dict[ci_lo, ci_hi, p, mde, sd]`; `two_way_estimates(diffs, line_index,
drug_index, n_draws, seed)`; `holm(p) -> ndarray`; `planted_reliability_pool(R, n_pairs, n_genes, seed)` and the
other synthetic builders in `controls.py`.

- [ ] **score control:** at R = 0.7353 and 0.1503, the true change scores √R and a second noisy copy scores R,
  within tolerance, through `score_pairs` (`test_score_square_root_known_answer`); an independent prediction
  scores 0 (`test_score_independent_prediction_is_zero`).
- [ ] **null control:** over repeated synthetic grids, a contrast planted at its MDE is detected at 0.80 within
  Monte Carlo tolerance and a zero contrast at ≤ 0.05 (`test_null_detection_at_mde`, `test_null_false_positive_rate`);
  Holm on a hand example; redraws seeded.
- [ ] **fit control (known answer), both schemes:** a synthetic grid of the screen's size (50 × 107, 300 genes)
  with line-specific responses linear in a description (LOLO), or shared by chemically similar drugs and modulated
  by the line description (LODO), planted at twice that grid's MDE → the matching description's gain over the
  reference is within tolerance of the planted gain, its random stand-in's is not; with nothing planted every gain
  is within its MDE and λ sits at the top of its range (`test_fit_recovers_planted_line_response`,
  `test_fit_recovers_planted_drug_response`, `test_fit_null_shrinks_to_reference`).
- [ ] **split control:** a signature planted only in one unit's own answers is recovered by a leaky fit that keeps
  the held-out unit, and scores zero within MDE under the shipped splits, both schemes (`test_split_leaky_recovers_signature`,
  `test_split_shipped_does_not`).
- [ ] Gates; commit.

### Task 10 — Rounds, redraws, combine, figures, leakage records

**Files** `scripts/heldout_fit.py`, `scripts/heldout_redraws.py`, `scripts/heldout_combine.py`,
`src/fmharness/heldout/figures.py`, `src/fmharness/heldout/leakage.py`, `tests/test_heldout_pipeline.py`.
**Interfaces** `run_round(scheme, index, inputs) -> (scores, settings)`; `run_redraw_block(pair_scores, block,
draws_per_block=250, seed) -> DataFrame`; `combine(out_dir, cache) -> None`; one figure function per design §8
bullet, each taking the table it draws; `leakage_profiles(grid, drug_metadata) -> list[LeakageProfile]`.

- [ ] `test_staged_run_equals_one_process` on a 6-line × 8-drug fixture with fixture descriptions: 6 + 8 rounds,
  4 redraw blocks, combine → tables identical to an in-process run. A round whose outputs exist skips.
- [ ] Combine writes every table in the contracts section, the control tables (from `controls.py`, at the real
  grid's size and the real ceiling values), the figures (non-empty PNGs, one per design §8 bullet), leakage
  records validating against `LeakageProfile`, and `rung1_run.params.json` (git sha, arguments, seeds, input sha256s).
- [ ] Gates; commit.

### Task 11 — The Alpine job chain, and the run

**Order.** The data stages (grid, answers, DMSO cells, embeddings, descriptions) depend only on Tasks 1–6, so their
job scripts are written and submitted right after Task 6, and the data is ready by the time Task 10 lands. The fit,
redraw and combine jobs follow Task 10. Both halves live in this task's files.

**Files** `scripts/alpine/rung1_env.sh`, `rung1_grid.sbatch`, `rung1_answers.sbatch` (array 0-7), `rung1_answers_combine.sbatch`,
`rung1_dmso_cells.sbatch` (array 0-63%4), `rung1_dmso_combine.sbatch`, `rung1_embed.sbatch` (GPU array 0-2),
`rung1_descriptions.sbatch`, `rung1_fit.sbatch` (array 0-156: 0-49 lines, 50-156 drugs),
`rung1_redraws.sbatch` (array 0-7), `rung1_combine.sbatch`, `submit_rung1_chain.sh`; `tests/test_rung1_jobs.py`.

- [ ] Env as rung 0's: stack env on PATH, `PYTHONPATH=$REPO/src`, Hugging Face cache and token, fallback
  `pip install "rdkit>=2024.3,<2026"`, `RUNG1_CACHE`, `OUT_DIR`, and the array sizes in one place.
  CPU jobs `--partition=acpu --qos=cpu-normal`; memory in whole cores (3,840 MB); GPU
  `--partition=ah200 --qos=gpu-normal --gres=gpu:h200:1`. Test: each array's range matches the env constants;
  every job sources the env; every script is submitted by the chain with `--dependency=afterok`.
- [ ] Local synthetic dry run of the CLIs in order passes. Push; `ralpine switch rung1-held-out-prediction-work`;
  `ralpine update`; `ralpine ls`/`du` the inputs (pseudobulk table, checkpoints, rung 0 promoted table).
- [ ] Check queues (`ralpine run squeue -p acpu -t PENDING`, `ralpine run sinfo -p amem`); if `acpu` is deep and
  `amem` idle, submit the fit array packed onto `amem` (whole node, several rounds per task). Submit the chain;
  confirm each dependency with `ralpine jobinfo`. Monitor; resubmit failed indexes alone.
- [ ] Pull `OUT_DIR` tables and figures and every log (`results/rung1-held-out-prediction/logs/`). Record jobs,
  nodes, wall times and memory in `verification.md`.

### Task 12 — Verification battery

**Files** `scripts/verify_rung1.py`, `tests/test_verify_rung1.py`, `docs/tasks/rung1-held-out-prediction/verify.ipynb`,
`verification.md`.

- [ ] The script recomputes from the artifacts alone: grid counts and hashes; the ceiling table from rung 0's
  promoted table; each model's mean score and fraction; every comparison estimate from the per-pair scores; CIs,
  p-values and MDEs from the redraw tables; Holm from p; the two excluded pairs absent; figure tables present;
  params sidecar sha matches. Prints claim / recomputed / pass-fail; exits non-zero on any fail.
- [ ] Test runs it on the Task 10 fixture outputs; skips on a clean checkout without run outputs.
- [ ] `verify.ipynb`: the same claims recomputed inline (standard library and pandas only, no project imports),
  committed without outputs; final cell calls the script as a cross-check.
- [ ] `verification.md`: commands, outputs, job table, pointers to every artifact.

### Task 13 — Review, audit, summary (stop at gate 2)

- [ ] Code review of the branch against `design.md` and this plan (superpowers:requesting-code-review);
  `review.md` records findings and dispositions; fixes committed; a scoped re-run only if a fix changes a number.
- [ ] Audit per `docs/audit.md`: a fresh reader turns `design.md` into numbered claims, checks them and the
  reverse diff against the tree, records `audit_checksums.json` for every artifact read; one fix wave; re-audit
  of drift items only.
- [ ] `summary.ipynb` (committed without outputs): hypotheses; each step in order with its figure beside its
  table and control; the comparisons; conclusions; scripts touched. **Stop and ask Lucas to review (gate 2).**

## Commits during execution

One commit per task on `rung1-held-out-prediction-work`, pushed when Alpine needs the code. No run output is
staged. The topic rebuild onto `rung1-held-out-prediction` (design §10) happens after gate 2.
