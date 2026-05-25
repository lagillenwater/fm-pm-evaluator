#!/usr/bin/env python3
"""Download GDSC2 drug-response + CCLE/DepMap RNA-seq for the sarcoma subset.

Two open data sources, paired on COSMIC ID:

1. GDSC2 (Sanger Institute, release 8.5, dated 27Oct23). Fitted dose-response
   (AUC, IC50), compound metadata, cell-line metadata. Hosted on the public
   Sanger CDN; no auth required.

2. DepMap (Broad Institute, latest public release, ~quarterly). RNA-seq TPM
   matrix and model metadata mapping DepMap ACH-XXXXX IDs to COSMIC IDs.
   DepMap distributes through figshare collections that change per release
   (25Q1, 25Q2, 24Q4...). To grab the URLs:

   Go to https://depmap.org/portal/download/all/, pick the latest release,
   right-click each file below and copy the figshare ndownloader link, then
   paste into DEPMAP_FILES below. Files needed:

     - OmicsExpressionProteinCodingGenesTPMLogp1.csv   (log2(TPM+1) per gene)
     - Model.csv                                       (ACH <-> COSMIC, lineage, subtype)

Outputs land in data/raw/gdsc2_sarcoma/ with sha256 of each file recorded in
manifest.json.

Usage:
    python scripts/download/download_gdsc2_sarcoma.py
    python scripts/download/download_gdsc2_sarcoma.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "gdsc2_sarcoma"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

# GDSC2 release 8.5 (27Oct23) — verified live on the Sanger CDN.
# Bump these when Sanger publishes a new release.
GDSC2_BASE = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5"
GDSC2_FILES: list[tuple[str, str]] = [
    ("GDSC2_fitted_dose_response_27Oct23.xlsx",
     f"{GDSC2_BASE}/GDSC2_fitted_dose_response_27Oct23.xlsx"),
    ("screened_compounds_rel_8.5.csv",
     f"{GDSC2_BASE}/screened_compounds_rel_8.5.csv"),
    ("Cell_Lines_Details.xlsx",
     f"{GDSC2_BASE}/Cell_Lines_Details.xlsx"),
]

# DepMap files — release 26Q1 (canonical-id snapshot public-26q1-5bbf).
# URLs returned by /portal/data_page/api/data; depmap.org issues a 302 redirect
# to a signed Google Cloud Storage URL that's regenerated on each request, so
# the depmap.org URL itself is stable across runs.
# Stranded variant is the more recent (preferred) TPM derivation; the unstranded
# variant (no "Stranded" suffix) remains DepMap's "main file" for backward compat.
DEPMAP_RELEASE = "26Q1"
DEPMAP_BASE = "https://depmap.org/portal/download/api/download"
DEPMAP_FILES: list[tuple[str, str]] = [
    (
        "OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv",
        f"{DEPMAP_BASE}?file_name=downloads-by-canonical-id%2Fpublic-26q1-5bbf.27%2FOmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv&dl_name=OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv&bucket=depmap-external-downloads",
    ),
    (
        "Model.csv",
        f"{DEPMAP_BASE}?file_name=downloads-by-canonical-id%2Fpublic-26q1-5bbf.37%2FModel.csv&dl_name=Model.csv&bucket=depmap-external-downloads",
    ),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fm-pdo-evaluator/0.1"})
    with urllib.request.urlopen(req) as response, tmp.open("wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {
        "dataset": "gdsc2_sarcoma",
        "gdsc_release": "8.5",
        "depmap_release": DEPMAP_RELEASE,
        "files": {},
    }


def write_manifest(manifest: dict) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def fetch_with_sha(name: str, url: str, manifest: dict) -> None:
    dest = OUTPUT_DIR / name
    if dest.exists():
        actual = sha256_file(dest)
        recorded = manifest["files"].get(name, {}).get("sha256")
        if recorded and recorded == actual:
            print(f"[skip] {name} present with matching sha256")
            return
        if recorded and recorded != actual:
            sys.exit(
                f"[fail] {name} sha256 mismatch — expected {recorded}, got {actual}. "
                "Delete the file to re-download, or investigate upstream change."
            )
    print(f"[get ] {name} <- {url}")
    try:
        download(url, dest)
    except (HTTPError, URLError) as e:
        sys.exit(f"[fail] {name} download error: {e}")
    digest = sha256_file(dest)
    manifest["files"][name] = {
        "url": url,
        "sha256": digest,
        "bytes": dest.stat().st_size,
    }
    print(f"       sha256 {digest}")


def fetch_gdsc2(manifest: dict) -> None:
    for name, url in GDSC2_FILES:
        fetch_with_sha(f"gdsc2/{name}", url, manifest)


def fetch_depmap(manifest: dict) -> None:
    if not DEPMAP_FILES:
        print(
            "[warn] DEPMAP_FILES is empty. Visit\n"
            "       https://depmap.org/portal/download/all/\n"
            "       Pick the latest release, copy the figshare ndownloader URLs into "
            "DEPMAP_FILES in this script, then rerun."
        )
        return
    for name, url in DEPMAP_FILES:
        fetch_with_sha(f"depmap/{name}", url, manifest)


def verify_only(manifest: dict) -> None:
    bad = []
    for name, rec in manifest["files"].items():
        path = OUTPUT_DIR / name
        if not path.exists():
            bad.append(f"{name}: missing")
            continue
        actual = sha256_file(path)
        if actual != rec["sha256"]:
            bad.append(f"{name}: sha256 mismatch (expected {rec['sha256']}, got {actual})")
    if bad:
        print("\n".join(bad))
        sys.exit(1)
    print(f"[ok  ] {len(manifest['files'])} files verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="re-verify existing files; no download")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    if args.verify_only:
        verify_only(manifest)
        return

    fetch_gdsc2(manifest)
    fetch_depmap(manifest)
    write_manifest(manifest)
    print(f"[done] manifest written to {MANIFEST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
