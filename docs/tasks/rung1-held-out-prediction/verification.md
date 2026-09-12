# Rung 1 — verification record

**As of** 2026-09-11. Per `docs/PROCESS.md` §1 (Verify): the commands run, their output, and
where everything the run produces lives. The claim-by-claim recomputation is
[`verify.ipynb`](verify.ipynb) and its continuous-integration form `scripts/verify_rung1.py`;
this document records what was run, on what, and what came back.

## The run has not happened yet, and why

**Alpine's `/scratch/alpine` filesystem is not mounted.** After the data-center power event the
login and compute nodes come up with a bare stub directory in its place, so everything rung 1
caches there is unreachable: the working cache, rung 0's 89 GB pseudobulk DE table, the Hugging
Face cache, and the sci-Plex checkpoint. When the cluster briefly returned, the grid job failed
after 113 seconds with

```
mkdir: cannot create directory '/scratch/alpine': File exists
```

which is the unmounted filesystem, **not** a defect in the job scripts: the line that produced it
(`mkdir -p logs "$RUNG1_CACHE"`, `scripts/alpine/rung1_env.sh:47`) is correct and ran without
complaint in an earlier 64-minute job at the same commit. Nothing was resubmitted, and nothing in
this task worked around it.

The job chain is therefore written, tested and ready, and the table below is the plan it will be
run to. Job ids, nodes, wall times and peak memory are filled in from
`results/rung1-held-out-prediction/logs/` once storage returns and the chain is submitted.

| Stage | Script | Ask | Waits for | Status |
|---|---|---|---|---|
| grid, ceiling, crosswalk | `rung1_grid.sbatch` | 20 cores, 75G, 2 h | — | submitted, failed on the unmounted scratch (113 s) |
| answers scan (array 0-7) | `rung1_answers.sbatch` | 20 cores, 75G, 4 h | grid | not yet run |
| answers combine | `rung1_answers_combine.sbatch` | 16 cores, 60G, 2 h | the whole answers array | not yet run |
| DMSO cells (array 0-63%4) | `rung1_dmso_cells.sbatch` | 4 cores, 15G, 6 h | grid | not yet run |
| DMSO combine | `rung1_dmso_combine.sbatch` | 8 cores, 30G, 3 h | the whole DMSO array | not yet run |
| Stack embeddings (GPU array 0-2) | `rung1_embed.sbatch` | ah200, 8 cores, 64G, 4 h | DMSO combine | not yet run |
| descriptions | `rung1_descriptions.sbatch` | 8 cores, 30G, 3 h | DMSO combine | not yet run |
| **fit (array 0-156)** | `rung1_fit.sbatch` | 4 cores, 15G, 2 h | answers combine, embeddings, descriptions | not yet run |
| **redraws (array 0-7)** | `rung1_redraws.sbatch` | 2 cores, 7,680M, 2 h | the **whole** fit array | not yet run |
| **combine** | `rung1_combine.sbatch` | 4 cores, 15G, 3 h | the **whole** redraw array | not yet run |

Every CPU job's memory is a whole number of Alpine cores at 3,840 MB each (PROCESS §2), and the
three fit-stage jobs are sized from task 10's measurements: a round holds about 2.2 GB of answer
arrays plus `models.MAX_BYTES` (2 GiB) of working blocks, and the combine was measured at 8 GB
and about an hour, CPU only. The chain is submitted by
`scripts/alpine/submit_rung1_chain.sh --stage data` and then `--stage fit`, each dependency read
back with `ralpine jobinfo`.

## Commands run locally, and what they returned

The measurement code is exercised end to end on a synthetic **6-line × 8-drug** fixture run —
the 14 rounds, all 8 redraw blocks and the combine, each through its own command line — which is
what the battery and both notebooks were run against while the cluster is down.

```
uv run pytest -o addopts="" -q                    462 passed, 9 skipped, 7 warnings (3 m 31 s)
uv run pytest tests/test_rung1_jobs.py            71 passed, 8 skipped
uv run pytest tests/test_verify_rung1.py \
              tests/test_heldout_pipeline.py \
              -W error                            48 passed, 1 skipped
uv run ruff check <scripts, tests, notebooks>     All checks passed!
uv run ruff format --check <the same>             already formatted
uv run pyright                                    0 errors, 0 warnings, 0 informations
uv run pyright scripts/verify_rung1.py            0 errors, 0 warnings, 0 informations
uv run --python 3.10 ... heldout_smoke_import.py  python 3.10.20 / ok
```

The 7 warnings are task 4's anndata `ImplicitModificationWarning`s in `tests/test_heldout_cells.py`
and pre-date this task; the 9 skips are the tests that need artifacts this tree does not have
(the committed-run check) or rules that do not apply to a job (the DuckDB overhead rule, which
binds only the jobs that scan the DE table).

**The battery, against the fixture run:**

```
uv run python scripts/verify_rung1.py --task-dir <fixture out> --cache <fixture cache>
38 / 38 checks pass (8 skipped, 46 total)          exit status 0
```

The 8 skips are stated, not waived: three claims hold only of the design's 50 × 107 grid (its
5,350 pairs, its candidate component counts, and the two sci-Plex pairs, which need A549 to be in
the grid), three need rung 0's promoted table to contain this grid's pairs, and two need the grid
record's hashes, which the fixture does not write. `tests/test_verify_rung1.py` exercises each of
those paths on inputs that do have them, and requires the battery to **fail** when a mean score,
a comparison estimate, a Holm adjustment, a removed pair or a pinned input's checksum is moved.

**Both reviewer notebooks, executed end to end** (`jupyter nbconvert --execute`, against the same
fixture run):

```
verify.ipynb    70 PASS, 0 FAIL, 0 errors; its cross-check of the battery exits 0
summary.ipynb   0 errors, all five figures displayed
```

and again against an **empty** task directory, which is what a fresh clone looks like:

```
verify.ipynb    0 errors; every cell reports "artifact not present yet"; the battery exits 2
summary.ipynb   0 errors; "run present: False"
```

Both notebooks are committed **without outputs**, so the figures and numbers a reviewer sees are
the ones their own execution produced.

## Where everything is

- **Tables** (once the run lands, uncommitted until promotion):
  `docs/tasks/rung1-held-out-prediction/rung1_pair_scores.csv.gz`, `rung1_model_summary.csv`,
  `rung1_comparisons.csv`, `rung1_settings.csv`, the five `rung1_control_*.csv`,
  `rung1_grid.json`, `rung1_ceiling.csv`, `rung1_leakage_profiles.json`, and the build stages'
  `rung1_cells.csv`, `rung1_identity_match*.csv`, `rung1_identity_grid_*.csv`,
  `rung1_weights_check.json`.
- **Figures**: `docs/tasks/rung1-held-out-prediction/figures/01_build.png` … `05_null.png`, one
  per design §8 step, each drawn from a table in the same folder.
- **Parameter sidecar**: `rung1_run.params.json` — the producing commit, the job id, every seed
  (including the redraws' base seed 7000), the realized candidate component counts, and the
  sha256 of every input the run read.
- **Per-round and per-block files**: `scores_{scheme}_{i}.parquet`, `settings_{scheme}_{i}.csv`
  and `redraws_{b}.parquet` in the run's scratch cache, each with a `.done.json` completion
  record holding its sha256. The battery reads them when `--cache` points at that cache.
- **Rung 0's inherited ceiling**: `results/rung0-assay-reliability/rung0_per_pair_r.csv` and
  `rung0_dose_strata.csv`, both committed, from which `rung1_ceiling.csv` is rebuilt.
- **Logs**: `results/rung1-held-out-prediction/logs/` once the chain has run.

## Open

- The cluster run itself, and with it every number in the tables above. Nothing is promoted, and
  nothing in this record is a result: it says what was checked about the code and the documents,
  on a synthetic run, while Alpine's storage is down.
- The whole-branch code review and the fresh-reader audit (`docs/audit.md`) are dispatched
  separately and are not part of this record.
