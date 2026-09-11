#!/usr/bin/env bash
# Submit rung 1's data stages as a dependency chain:
#
#   grid -> answers array -> answers combine -\
#        \-> dmso array    -> dmso combine    -> embed array
#                                              -> descriptions
#
# The fit, redraw and combine stages (task 11, part B) are `--stage fit`, added once Task 10
# lands; this script only knows `--stage data`.
#
# Run from the repository root on the LOCAL machine; every submission goes through ralpine so
# the boundary it enforces holds (PROCESS section 2). Each job id is read from sbatch's own
# output and passed as --dependency=afterok to the next, then the dependency is read back with
# `ralpine jobinfo` -- a chain that silently drops its dependency runs immediately and out of
# order (PROCESS section 2, "Chained jobs").
#
#   scripts/alpine/submit_rung1_chain.sh --stage data                # all six data jobs
#   scripts/alpine/submit_rung1_chain.sh --stage data --from answers # grid already done
#   scripts/alpine/submit_rung1_chain.sh --stage data --from dmso    # grid + answers done
#   scripts/alpine/submit_rung1_chain.sh --stage data --from embed   # everything but embed +
#                                                                    # descriptions done
set -euo pipefail

RALPINE="$(dirname "${BASH_SOURCE[0]}")/ralpine"
STAGE=""
FROM="grid"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="${2:?--stage needs a value: data}"
      shift 2
      ;;
    --from)
      FROM="${2:?--from needs a stage: grid, answers, dmso, embed}"
      shift 2
      ;;
    *)
      echo "unrecognized argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$STAGE" != "data" ]]; then
  echo "only --stage data is implemented here; the fit stage is task 11 part B" >&2
  exit 1
fi
case "$FROM" in
  grid|answers|dmso|embed) ;;
  *) echo "--from must be one of: grid, answers, dmso, embed" >&2; exit 1 ;;
esac

job_id() { grep -oE '[0-9]+$' <<<"$1" | tail -1; }
check_dependency() {
  local id="$1" want="$2"
  local dep
  dep="$("$RALPINE" jobinfo "$id" | grep -oE 'Dependency=[^ ]+' || true)"
  if [[ "$dep" != *"$want"* ]]; then
    echo "job $id was submitted without its dependency ($dep, wanted $want); cancel it" >&2
    exit 1
  fi
  echo "  job $id: $dep"
}

GRID_DEP=""
if [[ "$FROM" == "grid" ]]; then
  out="$("$RALPINE" submit scripts/alpine/rung1_grid.sbatch)"
  GRID="$(job_id "$out")"
  echo "grid:              job $GRID"
  GRID_DEP="--dependency=afterok:$GRID"
fi

if [[ "$FROM" == "grid" || "$FROM" == "answers" ]]; then
  out="$("$RALPINE" submit scripts/alpine/rung1_answers.sbatch ${GRID_DEP:+"$GRID_DEP"})"
  ANSWERS="$(job_id "$out")"
  echo "answers array:     job $ANSWERS"
  [[ -n "$GRID_DEP" ]] && check_dependency "$ANSWERS" "afterok:$GRID"

  out="$("$RALPINE" submit scripts/alpine/rung1_answers_combine.sbatch "--dependency=afterok:$ANSWERS")"
  ANSWERS_COMBINE="$(job_id "$out")"
  echo "answers combine:   job $ANSWERS_COMBINE"
  check_dependency "$ANSWERS_COMBINE" "afterok:$ANSWERS"
fi

DMSO_COMBINE=""
if [[ "$FROM" == "grid" || "$FROM" == "answers" || "$FROM" == "dmso" ]]; then
  out="$("$RALPINE" submit scripts/alpine/rung1_dmso_cells.sbatch ${GRID_DEP:+"$GRID_DEP"})"
  DMSO="$(job_id "$out")"
  echo "dmso array:        job $DMSO"
  [[ -n "$GRID_DEP" ]] && check_dependency "$DMSO" "afterok:$GRID"

  out="$("$RALPINE" submit scripts/alpine/rung1_dmso_combine.sbatch "--dependency=afterok:$DMSO")"
  DMSO_COMBINE="$(job_id "$out")"
  echo "dmso combine:      job $DMSO_COMBINE"
  check_dependency "$DMSO_COMBINE" "afterok:$DMSO"
fi

COMBINE_DEP=""
[[ -n "$DMSO_COMBINE" ]] && COMBINE_DEP="--dependency=afterok:$DMSO_COMBINE"

out="$("$RALPINE" submit scripts/alpine/rung1_embed.sbatch ${COMBINE_DEP:+"$COMBINE_DEP"})"
EMBED="$(job_id "$out")"
echo "embed array:       job $EMBED"
[[ -n "$COMBINE_DEP" ]] && check_dependency "$EMBED" "afterok:$DMSO_COMBINE"

out="$("$RALPINE" submit scripts/alpine/rung1_descriptions.sbatch ${COMBINE_DEP:+"$COMBINE_DEP"})"
DESCRIPTIONS="$(job_id "$out")"
echo "descriptions:      job $DESCRIPTIONS"
[[ -n "$COMBINE_DEP" ]] && check_dependency "$DESCRIPTIONS" "afterok:$DMSO_COMBINE"

echo "watch with: scripts/alpine/ralpine sq"
