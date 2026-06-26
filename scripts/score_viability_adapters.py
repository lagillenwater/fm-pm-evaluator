"""Compare delta sources x viability adapters against the Soragni AUC target.

The generation axis must be fair: the readout adapters (szalai/xgboost supervised on
real L1000 deltas vs GDSC2 AUC; hallmark unsupervised) are applied to EVERY delta
source, not just Stack's. Sources:

  - ``additive`` (always): each drug's mean real L1000 delta, applied to every organoid
    (organoid-independent) -- the generation analogue of the drug-mean baseline. The
    floor Stack must beat: it carries the drug main effect but no organoid x drug
    interaction.
  - ``stack`` (when --generated-dir is given): Stack-generated organoid-specific deltas.

Every (source, adapter) cell is scored against the real Soragni AUC with the same
global / interaction rho + within-drug label-permutation null, so Stack's generated
delta is compared head-to-head against the additive baseline under each readout.
Run on Alpine (needs the L1000 .gctx for the training cohort and additive source).

  PYTHONPATH=src python scripts/score_viability_adapters.py --l1000-dir . \\
      --gctx GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx \\
      --generated-dir generated_rich/ --baseline data/reference/stack_input_soragni.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.adapters import ALL_METHODS, build_adapters
from fmharness.controls import permute_within_drug
from fmharness.data.loaders import load_tranche
from fmharness.evaluation import (
    build_sample_design,
    global_spearman,
    interaction_rho,
    regret_norm_at_k,
)
from fmharness.l1000 import (
    build_additive_deltas,
    build_generated_deltas,
    build_l1000_gdsc_pairs,
    build_learned_deltas,
    soragni_pert_map,
)
from fmharness.signatures import load_hallmark

SEED = 0


def _score(preds: pd.DataFrame, n_perm: int) -> tuple[float, float, float, dict[int, float]]:
    """global rho, interaction rho, within-drug label-permutation p, regret@k dict."""
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
    return gl, it, float(np.mean(null >= it)), regret_norm_at_k(preds)


def _read_baseline(path: Path) -> pd.DataFrame:
    """Soragni baseline AnnData -> DataFrame (organoid x gene symbol)."""
    a = ad.read_h5ad(path)
    x = a.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return pd.DataFrame(
        np.asarray(x, dtype=np.float64),
        index=pd.Index([str(o) for o in a.obs_names]),
        columns=pd.Index([str(g) for g in a.var_names]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1000-dir", default=".")
    ap.add_argument("--gctx", required=True)
    ap.add_argument(
        "--generated-dir",
        default=None,
        help="Stack-generated per-drug .h5ad dir; omit to score the additive baseline only",
    )
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

    _, design = build_sample_design(load_tranche("sarcoma", repo), "tumor", "viability")

    # train cohort: real L1000 deltas -> GDSC2 AUC (for the supervised adapters and the
    # additive baseline). Keep the full delta for the additive per-drug mean; fit the
    # supervised adapters on the subset that has a GDSC2 AUC label.
    tr_delta, tr_key, dg, tr_base = build_l1000_gdsc_pairs(
        repo,
        Path(args.l1000_dir),
        args.gctx,
        time=args.time,
        chunk=args.chunk,
        treated_cap=args.treated_cap,
        dmso_cap=args.dmso_cap,
    )
    tr_via_all = tr_key.merge(dg.rename(columns={"y": "_y"}), on=["patient", "drug"], how="left")[
        "_y"
    ].to_numpy()
    ok = ~np.isnan(tr_via_all)
    tr_delta_fit, tr_via = tr_delta[ok], tr_via_all[ok]

    # delta sources, fed through the SAME readout adapters:
    #   additive  -- drug-mean L1000 delta (organoid-independent floor)
    #   pca / nmf -- learned organoid-specific delta predictors (need the Soragni baseline)
    #   stack     -- Stack-generated organoid-specific delta (when --generated-dir given)
    patients = sorted(str(p) for p in design["patient"].unique())
    base_path = Path(args.baseline) if Path(args.baseline).is_absolute() else repo / args.baseline
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "additive": build_additive_deltas(tr_delta, tr_key, patients)
    }
    if base_path.exists():
        soragni_base = _read_baseline(base_path)
        for reducer in ("pca", "nmf"):
            sources[reducer] = build_learned_deltas(
                tr_base, tr_delta, tr_key, soragni_base, patients, reducer=reducer
            )
    else:
        print(f"(skipping pca/nmf sources: baseline {base_path} not found)")
    if args.generated_dir:
        sources["stack"] = build_generated_deltas(
            Path(args.generated_dir), base_path, soragni_pert_map(repo)
        )

    out: list[dict[str, object]] = []
    for src_name, (sdelta, skey) in sources.items():
        common = tr_delta_fit.columns.intersection(sdelta.columns)
        tr_x, sx = tr_delta_fit[common], sdelta[common]
        print(
            f"[{src_name}] train {len(tr_x)} pairs | source {len(sx)} pairs | "
            f"{len(common)} shared genes | methods {methods}"
        )
        for adapter in build_adapters(methods, signatures=sigs):
            if adapter.supervised:
                adapter.fit(tr_x, tr_via)
            sens = adapter.predict(sx)
            merged = pd.DataFrame(
                {
                    "patient": skey["patient"].to_numpy(),
                    "drug": skey["drug"].to_numpy(),
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
            gl, it, pv, regret = _score(preds, args.n_permutations)
            out.append(
                {
                    "source": src_name,
                    "method": adapter.name,
                    "global": round(gl, 3),
                    "interaction": round(it, 3),
                    "p_label": round(pv, 3),
                    "regret@1": round(regret.get(1, float("nan")), 3),
                    "regret@3": round(regret.get(3, float("nan")), 3),
                    "n": len(preds),
                }
            )

    print("\n=== delta source x viability adapter vs Soragni AUC ===")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
