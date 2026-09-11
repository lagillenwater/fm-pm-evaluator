# Project state

**Reads against** [`docs/SPEC.md`](SPEC.md).
The spec says what each rung must establish and what a passing result means; this document says where each one stands.
It carries no history: a rung's result belongs here, how it came to be belongs in git and in that rung's spec.

**As of** 2026-09-11.

A number not carried here with its provenance record is not evidence.
Promotion means a result in `results/<task-slug>/` with a `<result>.provenance.json` beside it recording the commit, job and inputs that produced it — project rule 1.
Rung 0 is promoted on branch `rung0-assay-reliability` and closes when that branch merges: the dose-fixed run's ceilings are declared per dose, each clearing mismatched-pair floors drawn at its own dose, with provenance naming the commit the run was made at. The dose-pooled promotion of 2026-09-02 is withdrawn. No rung is closed on `main`.

---

## Ladder status

| Rung | What the spec requires | Status |
|---|---|---|
| 0 — assay reliability | Two reproducibility ceilings clearing their nulls at the assay's full extent — all genes, and each condition's responders — with replicate noise decomposed into plate and cell-sampling parts | **Promoted on branch, closes on merge.** Dose held fixed, plates split alternately ([design](tasks/rung0-assay-reliability/design.md) · [summary](tasks/rung0-assay-reliability/summary.ipynb) · [verification](tasks/rung0-assay-reliability/verification.md) · [result](../results/rung0-assay-reliability/rung0_dose_strata.csv)). The ceilings are the dose-level ones, each against floors drawn at its own dose: responders **0.577** at 5 uM (Spearman-Brown **0.732**, over 4,635 triples), 0.136 at 0.05 uM, 0.035 at 0.5 uM (narrowly, p 0.026); all genes 0.081, 0.029 and 0.024. All six clear their floors. The mean over all 7,641 replicated triples (0.065 all genes, 0.430 responders) is reported, not divided by. Noise: the published standard errors exceed the variance across plates for all genes, so no plate component is detectable there |
| 1 — held-out line or drug | Head-to-head comparisons of line descriptions (expression, PCA, NMF, three Stack versions) against the line-blind reference and random stand-ins, hiding one line or one drug at a time, at 5 uM, scored as a fraction of √SB from rung 0 | **In progress** on branch `rung1-held-out-prediction-work` ([design](tasks/rung1-held-out-prediction/design.md), approved 2026-09-11). No result yet |
| 2 — bulk read by a single-cell model | A synthesised population landing near the same material's real single cells, clearing a mismatched-line null | Not started |
| 3 — cross-platform | Retention separable from a scrambled-line control | Not started |
| 4 — GDSC2 viability | Interaction above zero after correction, against the screen-agreement ceiling | Not started |
| 5 — organoid viability | The transfer number, on a frozen embargoed holdout | Not started |
| 6 — prospective | A registered prediction holding on organoids screened afterwards | Not started |

A rung closes when its result is promoted with provenance, this table records it, and the project-rule tests for the steps it touches pass.


## What this repository holds today

| Present | Consequence |
|---|---|
| Schema, determinism and adapter scaffolding, with tests | The apparatus a rung is added to exists; nothing here yet produces a measurement |
| One promoted result, `results/rung0-assay-reliability/` | Rung 0 is the first work held to the spec's rules. The number is provisional and its provenance record says why; the rung is not closed |
| `docs/adapter_contract.md` and `docs/environment.md`, predating this spec | Neither has been reconciled against it. The rung that first depends on either brings it into line rather than a sweep that touches everything at once |

## Where things live

- **Results** `results/<task-slug>/<result>.csv` with `<result>.provenance.json` beside it. No provenance record, no evidence.
- **Figures** produced by a run into `docs/tasks/<slug>/figures/`, never drawn by hand, each drawn from a committed table and shown beside its control; declared per step in that task's `design.md`, pointed at from `verification.md`, and walked through in `summary.ipynb`. A figure the project cites is promoted alongside its table.
- **The reviewer's two notebooks** `summary.ipynb` explains the finding step by step with its figures; `verify.ipynb` recomputes every promoted claim inline from the committed artifacts. Both are committed without outputs.
- **Audits** `docs/tasks/<slug>/audit.md`, following the repository standard in [`docs/audit.md`](audit.md).
- **Rung and task specs** `docs/tasks/<slug>/design.md`, one folder per task, arriving with the work it specifies.
- **Decisions** dated and appended to the bottom of the task's own `design.md` and `plan.md`, so a reversal travels with the document it reverses.
- **Rules and their tests** `docs/SPEC.md` and `tests/test_project_rules.py`.
- **The summary a reader opens first** `README.md`, which restates this document's status line and the ladder from `docs/SPEC.md`. It moves whenever either does.
