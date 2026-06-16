"""Compare viability adapters on Stack-generated deltas, against the Soragni AUC target.

The supervised adapters (szalai, xgboost) are fit on real L1000 treated-minus-DMSO
deltas vs GDSC2 AUC; the unsupervised one (hallmark) needs no fit. Each is applied to
Stack's generated Soragni deltas and scored against the real Soragni AUC with the same
global / interaction rho + within-drug label-permutation null. ``--methods`` selects the
adapters (default: all three). Run on Alpine (needs the L1000 .gctx for the training cohort).

  PYTHONPATH=src python scripts/score_viability_adapters.py --l1000-dir . \\
      --gctx GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx \\
      --generated-dir generated_rich/ --baseline data/reference/stack_input_soragni.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fmharness.adapters import ALL_METHODS, build_adapters
from fmharness.controls import permute_within_drug
from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design, global_spearman, interaction_rho
from fmharness.l1000 import build_generated_deltas, build_l1000_gdsc_pairs, soragni_pert_map
from fmharness.signatures import load_hallmark

SEED = 0


def _score(preds: pd.DataFrame, n_perm: int) -> tuple[float, float, float]:
    """global rho, interaction rho, within-drug label-permutation p (vs interaction)."""
    gl = global_spearman(preds)
    it = interaction_rho(preds, "y_pred")
    null = np.array(
        [
            interaction_rho(
                preds.assign(
                    y_true=permute_within_drug(
                        preds["drug"], preds["y_true"], np.random.default_rng(SEED + 1 + b)
                    )
                ),
                "y_pred",
            )
            for b in range(n_perm)
        ]
    )
    return gl, it, float(np.mean(null >= it))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-dir", default=".")
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--generated-dir", required=True)
    ap.add_argument("--baseline", default="data/reference/stack_input_soragni.h5ad")
    ap.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help="comma-separated subset of hallmark,szalai,xgboost",
    )
    ap.add_argument(
        "--signatures",
        choices=["curated", "hallmark"],
        default="hallmark",
        help="gene sets for the hallmark adapter",
    )
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--time", type=float, default=24.0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--treated-cap", type=int, default=8)
    ap.add_argument("--dmso-cap", type=int, default=60)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sigs = (
        load_hallmark(repo / "data/static/hallmark_signatures.gmt")
        if "hallmark" in methods
        else None
    )

    # target: Stack-generated Soragni deltas + real Soragni AUC
    _, design = build_sample_design(load_tranche("sarcoma", repo), "organoid", "viability")
    base_path = Path(args.baseline) if Path(args.baseline).is_absolute() else repo / args.baseline
    tgt_delta, tgt_key = build_generated_deltas(
        Path(args.generated_dir), base_path, soragni_pert_map(repo)
    )

    # train cohort: real L1000 deltas -> GDSC2 AUC (for the supervised adapters)
    tr_delta, tr_key, dg = build_l1000_gdsc_pairs(
        repo,
        Path(args.l1000_dir),
        args.gctx,
        time=args.time,
        chunk=args.chunk,
        treated_cap=args.treated_cap,
        dmso_cap=args.dmso_cap,
    )
    tr_via = tr_key.merge(dg.rename(columns={"y": "_y"}), on=["patient", "drug"], how="left")[
        "_y"
    ].to_numpy()
    ok = ~np.isnan(tr_via)
    tr_delta, tr_via = tr_delta[ok], tr_via[ok]

    common = tr_delta.columns.intersection(tgt_delta.columns)
    tr_x, tgt_x = tr_delta[common], tgt_delta[common]
    print(
        f"train {len(tr_x)} pairs | target {len(tgt_x)} pairs | {len(common)} shared genes "
        f"| methods {methods}"
    )

    out: list[dict[str, object]] = []
    for adapter in build_adapters(methods, signatures=sigs):
        if adapter.supervised:
            adapter.fit(tr_x, tr_via)
        sens = adapter.predict(tgt_x)
        merged = pd.DataFrame(
            {
                "patient": tgt_key["patient"].to_numpy(),
                "drug": tgt_key["drug"].to_numpy(),
                "_sens": sens,
            }
        ).merge(design.rename(columns={"y": "y_true"}), on=["patient", "drug"], how="inner")
        preds = pd.DataFrame(
            {
                "patient": merged["patient"],
                "drug": merged["drug"],
                "y_true": merged["y_true"].to_numpy(),
                "y_pred": -merged["_sens"].to_numpy(),
            }
        )
        gl, it, pv = _score(preds, args.n_permutations)
        out.append(
            {
                "method": adapter.name,
                "global": round(gl, 3),
                "interaction": round(it, 3),
                "p_label": round(pv, 3),
                "n": len(preds),
                "citation": adapter.citation,
            }
        )

    print("\n=== viability adapters: Stack-generated deltas vs Soragni AUC ===")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
