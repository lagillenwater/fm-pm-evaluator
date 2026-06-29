"""Viability bridge: turn Stack-generated treated profiles into an AUC prediction.

Stack-Large-Aligned generates, per context drug, the *treated* expression of each
Soragni organoid. It does not output viability, so we bridge generated expression to
drug response with a fixed transcriptional readout:

  1. delta(s, d) = generated_treated(s, d) - baseline(s)      per organoid, drug
  2. z-score each gene's delta across all (organoid, drug) pairs
  3. sensitivity = direction-signed mean over a death / proliferation signature
       apoptosis / p53 up under a working drug -> sensitive
       proliferation down under a working drug -> sensitive
  4. score predicted sensitivity against the real Soragni AUC with the same
     interaction / within-drug / global metrics + permutation null used everywhere.

Sign convention: y_pred = -sensitivity (AUC-like), so a POSITIVE interaction rho
means the signature tracks real drug response.

  uv run python scripts/bridge_generated_to_auc.py --generated-dir generated/ \\
      --baseline data/reference/stack_input_soragni.h5ad --signatures hallmark
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fmharness.data.loaders import load_tranche
from fmharness.evaluation import build_sample_design
from fmharness.l1000 import build_generated_deltas, soragni_pert_map
from fmharness.signatures import load_hallmark, score_signatures

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-dir", required=True, help="dir of per-drug generated .h5ad")
    ap.add_argument("--baseline", default="data/reference/stack_input_soragni.h5ad")
    ap.add_argument("--n-permutations", type=int, default=1000)
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
    _, design = build_sample_design(load_tranche("sarcoma", repo), "tumor", "viability")
    p2s = soragni_pert_map(repo)
    print(f"  {len(p2s)} L1000 pert_id -> Soragni drug mappings")

    base_path = Path(args.baseline) if Path(args.baseline).is_absolute() else repo / args.baseline
    delta, key = build_generated_deltas(Path(args.generated_dir), base_path, p2s)
    res = score_signatures(
        delta, key, design, signatures=sigs, n_perm=args.n_permutations, seed=SEED
    )
    print(f"\ngenerated readout vs Soragni viability ({len(key)} organoid x drug)")
    print("PASS only if interaction > rnd_p95 (negative control) and p_label is small\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
