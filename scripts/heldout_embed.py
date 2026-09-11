"""Task 6 of rung 1: Stack embeddings per line and per half (Alpine GPU; design.md sections 4, 8).

Each of the three Stack versions (base ``bc_large.ckpt``, the released cytokine-aligned
``bc_large_aligned.ckpt``, and our sci-Plex drug fine-tune) describes a grid line by the
average Stack embedding of its untreated (DMSO) cells, from task 4's cache
(``cells/line_{i}.h5ad``). This script loads one checkpoint once, embeds every grid line's
cells with it, and writes:

* ``embedding_{version}_cells.npz`` (cache) -- per-cell embeddings, ``lines``, ``keys``,
  ``halves``, rows in each line file's own order, lines in grid order.
* ``embedding_{version}.parquet`` / ``embedding_{version}_halves.parquet`` (cache) -- per-line
  and per-(line, half) mean embeddings (``line_means``).
* ``rung1_identity_match_stack_{version}.csv`` / ``rung1_identity_grid_stack_{version}.csv``
  (out-dir) -- the build step's half-versus-half identity control for this description,
  standardized with the FULL per-line means' column moments, against a shuffled-label null
  built with the same aggregator (``fmharness.heldout.descriptions.group_means``).
* For ``--version drug`` only, ``rung1_weights_check.json`` (out-dir) -- a per-tensor sha256
  comparison of the drug fine-tune's encoder against the base checkpoint's (never the head),
  the build step's negative control (design.md section 8): the script exits non-zero if the
  encoders come out identical.

A strict ``state_dict`` load (never ``strict=False``) is tried on the checkpoint as given;
only if that fails is an encoder-only copy stripped into the cache
(``scripts/strip_ckpt_head.py``) and loaded strictly in its place. If the stripped copy still
does not load strictly, the script prints the missing/unexpected keys and exits non-zero.

Every output is followed by a ``<name>.done.json`` completion record
(``fmharness.heldout.records``); a rerun skips an output whose record is already valid. Torch,
anndata and ``stack`` imports stay inside the functions that need them, so this module (and
its pure ``line_means`` / ``encoders_differ``) imports locally without any of them.

    uv run python scripts/heldout_embed.py --version base \\
        --checkpoint stack-large/bc_large.ckpt \\
        --genelist stack-large/basecount_1000per_15000max.pkl \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --cache /scratch/alpine/$USER/rung1_cache \\
        --out-dir docs/tasks/rung1-held-out-prediction

    uv run python scripts/heldout_embed.py --version drug \\
        --checkpoint \\
            /scratch/alpine/$USER/sciplex_finetune/finetuned-epoch=5-val_loss=6.1078.ckpt \\
        --base-checkpoint stack-large/bc_large.ckpt \\
        --genelist stack-large/basecount_1000per_15000max.pkl \\
        --grid docs/tasks/rung1-held-out-prediction/rung1_grid.json \\
        --cache /scratch/alpine/$USER/rung1_cache \\
        --out-dir docs/tasks/rung1-held-out-prediction
"""

# pandas ships no PEP-561 type stubs in this environment; under strict mode that turns every
# call site into a cascade of reportUnknown* noise about *pandas'* types, not ours. Same
# suppression, same rationale as the rest of this project's pyright strict config where it
# touches scientific-Python packages -- the rules that check our own code stay on. ``torch``,
# ``anndata`` and ``stack`` are Alpine-only (never installed in this local environment, by
# design -- see the module docstring); their imports live inside the functions that need them,
# and reportMissingImports is off so pyright can still check the rest of this module strictly.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# So ``import strip_ckpt_head`` resolves whether this file is run directly (python already
# puts its own directory first) or loaded by path (tests, via importlib) -- both cases put
# this exact directory on sys.path rather than relying on the interpreter's default.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strip_ckpt_head import HEAD_SUFFIXES, head_keys, strip_checkpoint  # noqa: E402

from fmharness.heldout.descriptions import (  # noqa: E402
    RANDOM_SEEDS,
    group_means,
    identity_correlations,
    identity_match,
    shuffled_identity_correlations,
    shuffled_identity_null,
    standardize_apply,
    standardize_stats,
)
from fmharness.heldout.grid import load_grid  # noqa: E402
from fmharness.heldout.records import is_done, write_record  # noqa: E402

VERSIONS: tuple[str, ...] = ("base", "cytokine", "drug")


def _is_head_key(key: str) -> bool:
    return len(head_keys([key])) > 0


def line_means(
    cell_embeddings: np.ndarray, lines: np.ndarray, halves: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-line and per-(line, half) mean Stack embeddings, vectorized (via
    ``fmharness.heldout.descriptions.group_means``, no per-line loop).

    ``cell_embeddings`` is ``(n_cells, width)``; ``lines`` and ``halves`` label each row.
    Returns ``(per_line, per_half)``: ``per_line`` has index ``line`` (sorted distinct lines)
    and columns ``dim_0..dim_{width-1}``; ``per_half`` has a ``MultiIndex`` (``line``,
    ``half``) with the same columns.
    """
    width = cell_embeddings.shape[1]
    columns = [f"dim_{i}" for i in range(width)]

    line_labels, per_line_means = group_means(cell_embeddings, lines)
    per_line = pd.DataFrame(
        per_line_means, index=pd.Index(line_labels, name="line"), columns=columns
    )

    lines_arr = np.asarray(lines)
    halves_arr = np.asarray(halves)
    pairs = list(zip(lines_arr.tolist(), halves_arr.tolist(), strict=True))
    pair_labels, per_pair_means = group_means(cell_embeddings, pairs)
    pair_index = pd.MultiIndex.from_tuples(list(pair_labels), names=["line", "half"])
    per_half = pd.DataFrame(per_pair_means, index=pair_index, columns=columns)

    return per_line, per_half


def encoders_differ(
    state_a: Mapping[str, np.ndarray],
    state_b: Mapping[str, np.ndarray],
    keep: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Compare two encoder state dicts key-by-key: per-tensor sha256 (of bytes and shape) over
    the shared keys (after ``keep``, if given).

    Returns ``{"n_shared", "n_identical", "n_different", "identical_all", "only_in_a",
    "only_in_b"}``. ``identical_all`` is true only when there is at least one shared key and
    every shared key's array is byte-identical -- an empty shared set (e.g. disjoint keys)
    never counts as identical. Keys unique to one side are reported, not compared.
    """
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keep is not None:
        keys_a = {k for k in keys_a if keep(k)}
        keys_b = {k for k in keys_b if keep(k)}
    shared = sorted(keys_a & keys_b)
    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)

    n_identical = 0
    n_different = 0
    for key in shared:
        a = np.ascontiguousarray(state_a[key])
        b = np.ascontiguousarray(state_b[key])
        same = a.shape == b.shape and (
            hashlib.sha256(a.tobytes()).digest() == hashlib.sha256(b.tobytes()).digest()
        )
        if same:
            n_identical += 1
        else:
            n_different += 1

    return {
        "n_shared": len(shared),
        "n_identical": n_identical,
        "n_different": n_different,
        "identical_all": len(shared) > 0 and n_different == 0,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
    }


def _raw_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    """The raw ``state_dict`` of a checkpoint, torch tensors only (no model construction) --
    used only for the drug-vs-base weights check. Local ``torch`` import."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def load_stack_model(
    checkpoint_path: Path, cache: Path, model_class: str = "scShiftAttentionModel"
) -> Any:
    """Load a Stack checkpoint strictly. Tries ``checkpoint_path`` as given first; only if that
    raises does it strip generation-head keys into a cached encoder-only copy
    (``scripts/strip_ckpt_head.py``) and retry strictly on the stripped copy. Never loads with
    ``strict=False``: if the stripped copy still fails to load strictly, prints the missing/
    unexpected keys from the underlying error and exits the process non-zero. Local ``stack``
    import: only this function needs it.
    """
    from stack.model_loading import load_model_from_checkpoint

    try:
        return load_model_from_checkpoint(
            str(checkpoint_path), model_class=model_class, strict=True
        )
    except RuntimeError as first_error:
        stripped_path = cache / f"{checkpoint_path.stem}_encoder_only.ckpt"
        if not is_done(stripped_path):
            removed = strip_checkpoint(checkpoint_path, stripped_path)
            write_record(stripped_path, {"removed_keys": removed, "source": str(checkpoint_path)})
            print(
                f"stripped {len(removed)} head keys from {checkpoint_path} -> "
                f"{stripped_path}: {removed}"
            )
        try:
            return load_model_from_checkpoint(
                str(stripped_path), model_class=model_class, strict=True
            )
        except RuntimeError as second_error:
            print(
                f"strict load of {checkpoint_path} failed even after stripping head keys:",
                file=sys.stderr,
            )
            print(str(second_error), file=sys.stderr)
            print(
                f"(first attempt, before stripping, had failed with: {first_error})",
                file=sys.stderr,
            )
            sys.exit(1)


def embed_line(
    model: Any,
    line_path: Path,
    genelist_path: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """The Stack embedding of every cell in one line's cache file, plus its ``obs`` (for
    ``key``/``half``). Asserts one embedding per cell, naming ``line_path`` if that fails.
    Local ``anndata`` import: only this function needs it."""
    import anndata as ad

    adata = ad.read_h5ad(line_path)
    n_obs = adata.n_obs
    embeddings, _dataset_embeddings = model.get_latent_representation(
        adata_path=str(line_path),
        genelist_path=str(genelist_path),
        gene_name_col="feature_name",
        batch_size=batch_size,
        show_progress=False,
        num_workers=num_workers,
        random_state=seed,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != n_obs:
        raise AssertionError(
            f"{line_path}: expected {n_obs} cell embeddings (one per cell), "
            f"got {embeddings.shape[0]}"
        )
    return embeddings, cast(pd.DataFrame, adata.obs)


def compute_embeddings(
    version: str,
    checkpoint_path: Path,
    genelist_path: Path,
    grid_lines: tuple[str, ...],
    cache: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every grid line's cells embedded by one loaded checkpoint (loaded once). Returns
    ``(lines, keys, halves, embeddings)``, rows in each line file's own order, lines in grid
    order -- exactly ``embedding_{version}_cells.npz``'s arrays."""
    model = load_stack_model(checkpoint_path, cache)
    model_seed = RANDOM_SEEDS[f"stack_{version}"]

    lines_out: list[np.ndarray] = []
    keys_out: list[np.ndarray] = []
    halves_out: list[np.ndarray] = []
    embeddings_out: list[np.ndarray] = []
    for i, line in enumerate(grid_lines):
        line_path = cache / "cells" / f"line_{i}.h5ad"
        embeddings, obs = embed_line(
            model, line_path, genelist_path, batch_size, num_workers, model_seed
        )
        n = embeddings.shape[0]
        lines_out.append(np.full(n, line, dtype=object))
        keys_out.append(obs["key"].to_numpy())
        halves_out.append(obs["half"].to_numpy(dtype=np.int8))
        embeddings_out.append(embeddings)
        print(f"embedded {line} ({i + 1}/{len(grid_lines)}): {n} cells")

    return (
        np.concatenate(lines_out),
        np.concatenate(keys_out),
        np.concatenate(halves_out),
        np.concatenate(embeddings_out, axis=0),
    )


def identity_control(
    cell_embeddings: np.ndarray,
    lines_arr: np.ndarray,
    halves_arr: np.ndarray,
    per_line: pd.DataFrame,
    per_half: pd.DataFrame,
    version: str,
    n_shuffles: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The identity-match row and long-form correlation grid for ``stack_{version}``: the real
    statistic is ``identity_match`` on per-half means standardized with the FULL per-line
    means' column moments; the null is the shuffled-label null with ``group_means`` as the
    aggregator and the same standardizing ``describe``."""
    lines_sorted = np.asarray(per_line.index)
    mean_emb, std_emb = standardize_stats(per_line.to_numpy(dtype=np.float64))

    half0 = (
        per_half.xs(0, level="half")
        .reindex(lines_sorted, fill_value=0.0)
        .to_numpy(dtype=np.float64)
    )
    half1 = (
        per_half.xs(1, level="half")
        .reindex(lines_sorted, fill_value=0.0)
        .to_numpy(dtype=np.float64)
    )

    def describe(x: np.ndarray) -> np.ndarray:
        return standardize_apply(x, mean_emb, std_emb)

    std_half0 = describe(half0)
    std_half1 = describe(half1)
    real_share = identity_match(std_half0, std_half1)

    null = shuffled_identity_null(
        cell_embeddings, lines_arr, halves_arr, describe, n_shuffles, seed, aggregate=group_means
    )

    match_row = pd.DataFrame(
        [
            {
                "description": f"stack_{version}",
                "identity_share": real_share,
                "null_mean": float(null.mean()),
                "null_p99": float(np.quantile(null, 0.99)),
                "n_lines": len(lines_sorted),
                "n_shuffles": n_shuffles,
            }
        ]
    )

    real_grid = identity_correlations(std_half0, std_half1)
    n_lines = len(lines_sorted)
    real_long = pd.DataFrame(
        {
            "line_a": np.repeat(lines_sorted, n_lines),
            "line_b": np.tile(lines_sorted, n_lines),
            "r": real_grid.ravel(),
            "source": "real",
        }
    )

    shuffled_categories, shuffled_corr = shuffled_identity_correlations(
        cell_embeddings, lines_arr, halves_arr, describe, seed, aggregate=group_means
    )
    n_shuffled = len(shuffled_categories)
    shuffled_long = pd.DataFrame(
        {
            "line_a": np.repeat(shuffled_categories, n_shuffled),
            "line_b": np.tile(shuffled_categories, n_shuffled),
            "r": shuffled_corr.ravel(),
            "source": "shuffled",
        }
    )

    return match_row, pd.concat([real_long, shuffled_long], ignore_index=True)


def weights_check(drug_checkpoint: Path, base_checkpoint: Path) -> dict[str, Any]:
    """The drug fine-tune's encoder compared against the base checkpoint's, over encoder keys
    only (never the generation head): ``encoders_differ`` with ``keep`` excluding
    ``HEAD_SUFFIXES``-matched keys."""

    def keep(key: str) -> bool:
        return not _is_head_key(key)

    drug_state = {k: v.detach().cpu().numpy() for k, v in _raw_state_dict(drug_checkpoint).items()}
    base_state = {k: v.detach().cpu().numpy() for k, v in _raw_state_dict(base_checkpoint).items()}
    result = encoders_differ(drug_state, base_state, keep=keep)
    return {
        "drug_checkpoint": str(drug_checkpoint),
        "base_checkpoint": str(base_checkpoint),
        "head_suffixes": list(HEAD_SUFFIXES),
        **result,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", required=True, choices=VERSIONS)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--genelist", required=True, type=Path)
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="required for --version drug (the weights check)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="seed for the shuffled-identity null")
    ap.add_argument("--n-shuffles", type=int, default=200)
    args = ap.parse_args()

    if args.version == "drug" and args.base_checkpoint is None:
        ap.error("--base-checkpoint is required for --version drug")

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid(args.grid)

    cells_path = args.cache / f"embedding_{args.version}_cells.npz"
    per_line_path = args.cache / f"embedding_{args.version}.parquet"
    per_half_path = args.cache / f"embedding_{args.version}_halves.parquet"
    identity_path = args.out_dir / f"rung1_identity_match_stack_{args.version}.csv"
    grid_path = args.out_dir / f"rung1_identity_grid_stack_{args.version}.csv"
    weights_path = args.out_dir / "rung1_weights_check.json"

    need_embed = not (is_done(cells_path) and is_done(per_line_path) and is_done(per_half_path))
    need_identity = not (is_done(identity_path) and is_done(grid_path))
    need_weights = args.version == "drug" and not is_done(weights_path)

    cell_embeddings: np.ndarray | None = None
    lines_arr: np.ndarray | None = None
    halves_arr: np.ndarray | None = None
    keys_arr: np.ndarray | None = None

    if need_embed or need_identity:
        if is_done(cells_path):
            npz = np.load(cells_path, allow_pickle=True)
            lines_arr = npz["lines"]
            halves_arr = npz["halves"]
            cell_embeddings = npz["embeddings"]
            keys_arr = npz["keys"]
        else:
            lines_arr, keys_arr, halves_arr, cell_embeddings = compute_embeddings(
                args.version,
                args.checkpoint,
                args.genelist,
                grid.lines,
                args.cache,
                args.batch_size,
                args.num_workers,
            )

    if need_embed:
        assert cell_embeddings is not None
        assert lines_arr is not None
        assert halves_arr is not None
        assert keys_arr is not None
        np.savez(
            cells_path,
            lines=lines_arr,
            keys=keys_arr,
            halves=halves_arr,
            embeddings=cell_embeddings,
        )
        write_record(cells_path)
        print(f"wrote {cells_path}")

        per_line, per_half = line_means(cell_embeddings, lines_arr, halves_arr)
        per_line.to_parquet(per_line_path)
        write_record(per_line_path)
        print(f"wrote {per_line_path}")
        per_half.to_parquet(per_half_path)
        write_record(per_half_path)
        print(f"wrote {per_half_path}")
    else:
        print(f"{cells_path} already done, skipping")
        print(f"{per_line_path} already done, skipping")
        print(f"{per_half_path} already done, skipping")

    if need_identity:
        assert cell_embeddings is not None
        assert lines_arr is not None
        assert halves_arr is not None
        per_line = pd.read_parquet(per_line_path)
        per_half = pd.read_parquet(per_half_path)
        match_row, grid_table = identity_control(
            cell_embeddings,
            lines_arr,
            halves_arr,
            per_line,
            per_half,
            args.version,
            args.n_shuffles,
            args.seed,
        )
        match_row.to_csv(identity_path, index=False)
        write_record(identity_path)
        print(f"wrote {identity_path}")
        grid_table.to_csv(grid_path, index=False)
        write_record(grid_path)
        print(f"wrote {grid_path}")
    else:
        print(f"{identity_path} already done, skipping")
        print(f"{grid_path} already done, skipping")

    if need_weights:
        result = weights_check(args.checkpoint, args.base_checkpoint)
        weights_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        write_record(weights_path)
        print(f"wrote {weights_path}")
        if result["identical_all"]:
            print(
                "FAIL: the drug fine-tune's encoder is byte-identical to the base checkpoint's "
                "-- the fine-tune did not change the encoder",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.version == "drug":
        print(f"{weights_path} already done, skipping")


if __name__ == "__main__":
    main()
