"""Validate the viability-bridge readout on REAL data (Path-B go/no-go gate).

Before trusting Stack-generated deltas, check that the death/proliferation signature
predicts viability on *real* perturbation transcriptomes. We pair L1000 treated-minus-
DMSO deltas with GDSC2 AUC on shared (cell line, drug) pairs and score the signature
against viability with the same interaction / within-drug metrics used everywhere. If
the readout can't track real viability here, the Stack-generated deltas won't either.

Run on Alpine (needs the L1000 Level-3 matrix):
  PYTHONPATH=src python scripts/validate_bridge_readout.py --l1000-dir . \\
      --gctx GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fmharness.l1000 import build_l1000_gdsc_pairs
from fmharness.signatures import load_hallmark, score_signatures

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-dir", default=".")
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--time", type=float, default=24.0, help="L1000 pert_time to use")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument(
        "--chunk", type=int, default=2000, help="gctx columns read per chunk (caps peak memory)"
    )
    ap.add_argument(
        "--treated-cap", type=int, default=8, help="max treated wells averaged per (cell, drug)"
    )
    ap.add_argument("--dmso-cap", type=int, default=60, help="max DMSO wells averaged per cell")
    ap.add_argument(
        "--signatures",
        choices=["curated", "hallmark"],
        default="curated",
        help="curated death/proliferation set or the published MSigDB Hallmark sets",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    sigs = (
        load_hallmark(repo / "data/static/hallmark_signatures.gmt")
        if args.signatures == "hallmark"
        else None
    )
    delta, key, dg = build_l1000_gdsc_pairs(
        repo,
        Path(args.l1000_dir),
        args.gctx,
        time=args.time,
        chunk=args.chunk,
        treated_cap=args.treated_cap,
        dmso_cap=args.dmso_cap,
    )
    res = score_signatures(delta, key, dg, signatures=sigs, n_perm=args.n_permutations, seed=SEED)
    print(f"\nreadout vs GDSC2 viability  ({len(key)} cell-line x drug deltas)")
    print("PASS only if interaction > rnd_p95 (negative control) and p_label is small\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
