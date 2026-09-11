# Rung 0 — verification record

**As of** 2026-09-11. Per `docs/PROCESS.md` §1 (Verify): the commands run, their output, and
where everything the run produced lives. The claim-by-claim recomputation is `verify.ipynb`;
this document records that it was done, on what, and what came back.

## The run

The dose-fixed measurement, on code at `3121b65` and later (every stage's parameter sidecar
records its own `git_sha`), over the tranche `tahoe100m-pseudobulk-de.v1` on Alpine scratch,
as the chain the design declares — assign, slices, combine, permutation — with the slices
packed onto high-memory nodes on the day the general queue was deep.

| stage | job | partition | wall | peak RSS | produced |
|---|---|---|---|---|---|
| assign | 32366450 | acpu, 96G, engine 60G | 45 min 18 s | 83 GB | the split assignment (56,827 triples, 7,641 replicated) and the pool description |
| slices 0–10 | 32369084_0 | amem, whole node, 11 slices at once | 2 h 27 min | 838 GB | 11 cached slices |
| slices 11–21 | 32369084_1 | amem | 2 h 27 min | 836 GB | 11 cached slices |
| slices 22–31 | 32369084_2 | amem | 2 h 48 min | 758 GB | 10 cached slices |
| combine | 32369085 | amem, 256G | 8 min 23 s | 39 GB | every table and figure (first pass) |
| permutation, all genes, 500 | 32369086 | amem, 256G | 3 h 18 min to the all-gene summary | 39 GB | superseded: cancelled 1 h 19 min into the responder set, and the check was cut to 100 permutations (`decisions.md`, 2026-09-10) |
| combine (dose formatting fix, `3121b65`) | 32378445 | amem, 256G | 5 min 44 s | 40 GB | every table and figure, the committed set |
| permutation, both gene sets, 100 | 32378577 | amem, 256G, 3 h | 1 h 18 min (44 min all genes, 34 min responders) | 39 GB | `rung0_permutation_*.csv`, both `11_permutation_vs_bootstrap*.png` |
| combine (after 32378577) | 32378578 | amem, 256G | 6 min 53 s | 40 GB | superseded by the combine below |
| combine, final (per-dose floors, `8392557`) | 32379206 | amem, 256G | 5 min 51 s | 40 GB | the committed tables, with each dose level's own floors, p-values and MDEs and `rung0_null_draws_by_dose.csv`; `audit_checksums.json` records 47 artifacts |

The thirty-two slices read 4,089,820,780 rows once each and returned 174,343,417 scoreable
gene-conditions and 479,167,110 decomposable ones. Logs are in
`results/rung0-assay-reliability/logs/` (assign, the three packs, every slice, both combines).

Attempts that did not produce the result, kept for the record: 32341467 (assign, killed at 64
GB with the engine at 48 GB); 32341611 (sixteen 120 GB slices, billed as 32-core jobs, ran one
at a time and were cancelled); 32345592_0 (a thirty-second at 72 GB, killed at 75 GB); 32347952
(thirty-two slices sharing one spill directory — truncated temp files, segfaults, one engine OOM;
nothing from it reused); 32365763 (assign at a 40 GB engine, did not finish in 40 min);
32366451 and 32368430 (queued arrays superseded by the packed route). Each is explained in the
job scripts' comments and in `decisions.md`.

## Commands run locally, and what they returned

```
uv run pytest                          152 passed, 1 skipped (the committed-run test skips
                                       until the run is committed)
uv run ruff check . && ruff format --check . && pyright     clean
uv run python scripts/verify_rung0.py  77 / 79 checks pass, 0 skipped, 79 total
```

The two failures
are the same open item, not a defect of this run: `results/rung0-assay-reliability/` still holds
the dose-pooled promotion of 2026-09-02, whose table is not this run's and whose provenance
names `92407c1`, the promotion-time commit, where its own sidecar says `5192606`. It stays
until the decision to withdraw or re-promote it is made.

Both reviewer notebooks executed end to end against these artifacts (`jupyter nbconvert
--execute`): `verify.ipynb` printed 88 PASS and the same 2 FAIL; `summary.ipynb` ran without
error and reports hypotheses 1, 2 and 3 held; hypothesis 4 holds in the aggregate the design
states it in (variance across plates pooled over the responders' gene-conditions, 3.67, against
1.50 over all) and not triple by triple (the typical triple's responders are less variable than
its non-responders), and the notebook prints both readings.

## What the run says

| | all genes | responders |
|---|---|---|
| triples scored | 7,641 | 6,654 |
| split-half r, mean over triples | 0.065 | 0.430 |
| Spearman-Brown | 0.122 | 0.601 |
| equal-halves subset | 7,491 triples, 0.066 | 6,518, 0.437 |
| different-drug floor | 0.017 | 0.106 |
| same-drug, same-dose floor | 0.042 | 0.257 |
| p vs each floor (bootstrap) | 0.0005, 0.0005 | 0.0005, 0.0005 |
| permutation, 100: exact p | 0.0099 (the floor 100 permutations can give) | 0.0099 |
| permutation: observed against null, in null sds | 145 | 201 |
| design effect, different-drug stratum | 0.70 | 0.51 |

The declared ceilings are the dose-level rows (`decisions.md`, 2026-09-11), each read against
floors drawn from triples at its own dose (`rung0_dose_strata.csv`):

| dose (uM) | gene set | triples | mean r | Spearman-Brown | floor, different drug | floor, same drug and dose | p, p | clears both |
|---|---|---|---|---|---|---|---|---|
| 0.05 | all genes | 1,245 | 0.029 | 0.056 | 0.019 | 0.020 | 0.0005, 0.0005 | yes |
| 0.5 | all genes | 1,000 | 0.024 | 0.047 | 0.002 | 0.010 | 0.0005, 0.0005 | yes |
| 5.0 | all genes | 5,396 | 0.081 | 0.149 | 0.022 | 0.061 | 0.0005, 0.0005 | yes |
| 0.05 | responders | 1,129 | 0.136 | 0.239 | 0.065 | 0.068 | 0.0005, 0.0005 | yes |
| 0.5 | responders | 890 | 0.035 | 0.068 | -0.028 | 0.024 | 0.0005, 0.026 | yes, narrowly |
| 5.0 | responders | 4,635 | 0.577 | 0.732 | 0.161 | 0.357 | 0.0005, 0.0005 | yes |

Every dose-level ceiling clears both of its own floors. The margins differ by an order of
magnitude: responders at 5 uM sit 0.22 above their same-dose floor, responders at 0.5 uM 0.011
above theirs at p 0.026, the one ceiling a later rung would read against with little room. It
is significant with under 80% power: its MDE against the same-dose floor is 0.042, above its own
mean of 0.035, so a replication of the same size could easily fail to clear that floor. The
floors themselves move with dose as the correlations do, which is why pooled floors could not
stand in for them: the 5 uM responder same-dose floor is 0.357, well above the pooled 0.257.

Noise: pooled over all 174,564,006 gene-conditions the variance across plates (1.497) is
below the published squared standard error (2.858), so the plate component is zero and the
published errors are conservative for null genes. Pooled within the responders the share is
0.737, a number the design's selection rule inflates (responders are chosen on plate 0's
significance, which widens the plate-0-against-plate-1 difference by construction); it is
reported, not cited. Both tercile rankings rise (0.050 → 0.070 → 0.075; 0.061 → 0.061 →
0.073).

## Where everything is

- Tables: `docs/tasks/rung0-assay-reliability/rung0_*.csv{,.gz}` — the per-triple table
  (`rung0_per_pair_r.csv`) is the primary artifact; `rung0_dose_strata.csv` carries every
  candidate ceiling; `rung0_split_assignment.csv` is the split itself.
- Figures: `docs/tasks/rung0-assay-reliability/figures/01_build.png` … `10_dose.png`,
  `11_permutation_vs_bootstrap.png`, `11_permutation_vs_bootstrap_responder.png`, with
  `04_score.values.csv.gz` beside the score figure. `02_split.png` and both permutation figures
  were redrawn from the committed tables by `scripts/redraw_rung0_figures.py` after the run
  (`decisions.md`, 2026-09-11).
- Parameter sidecars: `rung0_reliability.params.json`, `rung0_noise_decomposition.params.json`,
  `rung0_permutation_summary.params.json`, `rung0_permutation_summary_responder.params.json` —
  each with the commit and job that produced it.
- Checksums: `audit_checksums.json`, one sha256 per artifact, recomputed by the battery.
- Held off the repository because of its size: `rung0_noise_per_gene.csv.gz` (73 MB, the
  two-million-row per-gene noise sample), sha256 `d0b506ff48ddfd67e3916b237dfe213c950b8d50a025c4945b32d949fa355f2a`, in the Alpine checkout's task folder.
  No promoted number is read from it; the battery's row-by-row identity and strata checks, which
  need it, run where it is present and skip where it is not.
- Logs: `results/rung0-assay-reliability/logs/`.

## Open

Nothing blocks promotion: the re-audit passed on its third pass (`audit.md`, "Re-audit"), and the
dose-pooled promotion of 2026-09-02 is withdrawn in the promotion commit (`decisions.md`,
2026-09-11).
