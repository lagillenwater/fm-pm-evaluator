"""Strip the ICL generation-head weights from an aligned Stack checkpoint (rung 1 task 6).

``stack``'s embedding path loads a checkpoint into ``StateICLModel`` (the encoder-only
architecture, no generation head), so the cytokine-aligned checkpoint -- which carries the
extra head weights (``query_pos_embedding`` and the ``cls`` MLP) -- fails a strict
``state_dict`` load with "Unexpected key(s)". Those weights are unused for embedding, so drop
them: the remaining encoder weights are exactly the aligned model's representation. Base
(unaligned) checkpoints and our sci-Plex fine-tune (also head-free) need no stripping;
``scripts/heldout_embed.py`` only strips a checkpoint that fails to load strictly as-is.

* ``head_keys`` -- the pure, testable key selection (matched by suffix, so any
  ``LightningModule`` prefix such as ``model.`` is handled): no torch import, so this
  function -- and so this module -- imports without torch installed.
* ``strip_checkpoint`` -- loads a checkpoint, removes its head keys, and saves the
  encoder-only copy. Torch is imported locally inside this function (and ``main``) only.

    python scripts/strip_ckpt_head.py --in bc_large_aligned.ckpt --out bc_large_aligned_encoder.ckpt
"""

# ``torch`` is Alpine-only (never installed in this local environment, by design -- see the
# module docstring); its import lives inside the functions that need it, and reportMissingImports
# is off here so pyright can still check the rest of this small module strictly. Same rationale
# as this project's other pyright suppressions where a dependency ships no usable stubs locally.
# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: The keys StateICLModel rejects (matched by suffix, so any LightningModule prefix such as
#: ``model.`` is handled).
HEAD_SUFFIXES: tuple[str, ...] = (
    "query_pos_embedding",
    "cls.0.weight",
    "cls.0.bias",
    "cls.2.weight",
    "cls.2.bias",
)


def head_keys(keys: Iterable[str]) -> list[str]:
    """The subset of ``keys`` that belong to the ICL generation head: those ending in (or
    exactly equal to) one of ``HEAD_SUFFIXES``. Preserves the input order; no torch import, so
    this can be tested (and used to filter a plain ``state_dict``) without torch installed.
    """
    return [k for k in keys if any(k == suffix or k.endswith(suffix) for suffix in HEAD_SUFFIXES)]


def strip_checkpoint(in_path: Path, out_path: Path) -> list[str]:
    """Load ``in_path``, remove its generation-head keys (see ``head_keys``), and save the
    encoder-only copy to ``out_path``. Returns the removed key names, in the checkpoint's own
    key order. Local ``torch`` import: only this function (and ``main``) needs it, so the
    module imports without torch installed.
    """
    import torch

    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    state_dict: dict[str, Any] = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    removed = head_keys(state_dict.keys())
    for key in removed:
        del state_dict[key]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["state_dict"] = state_dict
    else:
        ckpt = state_dict
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="aligned checkpoint (.ckpt)")
    ap.add_argument(
        "--out", required=True, type=Path, help="destination encoder-only checkpoint (.ckpt)"
    )
    args = ap.parse_args()

    removed = strip_checkpoint(args.inp, args.out)
    if not removed:
        print("no gen-head keys found -- checkpoint may already be encoder-only; copied as-is")
    print(f"wrote {args.out}; removed {len(removed)} gen-head keys: {removed}")


if __name__ == "__main__":
    main()


__all__ = ["HEAD_SUFFIXES", "head_keys", "strip_checkpoint"]
