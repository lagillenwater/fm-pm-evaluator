"""Fetch the MSigDB Hallmark gene sets used by the viability bridge.

Downloads four Hallmark sets from Enrichr's MSigDB_Hallmark_2020 library (free,
no login; same gene content as MSigDB -- Liberzon et al., Cell Systems 2015) and
writes them as a GMT under data/static/. Two mark a working drug's death trace
(p53 / apoptosis, induced; direction +1) and two mark proliferation (E2F / G2-M,
suppressed; direction -1). Using published sets removes any dependence on a
hand-curated signature.

  uv run python scripts/fetch_hallmark.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ENRICHR = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary"
    "?mode=text&libraryName=MSigDB_Hallmark_2020"
)
# Enrichr display name -> canonical MSigDB Hallmark name
WANT = {
    "p53 Pathway": "HALLMARK_P53_PATHWAY",
    "Apoptosis": "HALLMARK_APOPTOSIS",
    "E2F Targets": "HALLMARK_E2F_TARGETS",
    "G2-M Checkpoint": "HALLMARK_G2M_CHECKPOINT",
}


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    out = repo / "data/static/hallmark_signatures.gmt"
    txt = urllib.request.urlopen(ENRICHR, timeout=60).read().decode()
    found: dict[str, list[str]] = {}
    for line in txt.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] in WANT:
            found[WANT[parts[0]]] = [g for g in parts[2:] if g.strip()]
    missing = set(WANT.values()) - set(found)
    if missing:
        raise SystemExit(f"missing Hallmark sets from Enrichr: {sorted(missing)}")
    lines = [
        f"{name}\tMSigDB_Hallmark_2020\t" + "\t".join(found[name])
        for name in WANT.values()
    ]
    out.write_text("\n".join(lines) + "\n")
    for name in WANT.values():
        print(f"{name}: {len(found[name])} genes")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
