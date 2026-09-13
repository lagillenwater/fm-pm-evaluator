"""Pull the PANACEA release (Columbia CTD2, PLATE-seq drug perturbations) from Synapse.

Synapse project syn20968331. The token is read from ~/.synapse_token (never passed on the
command line or logged). Files land under --out; the file list with sizes is printed and
written beside them so the line/drug overlap with Tahoe can be judged before anything is read.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import synapseclient
import synapseutils


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", default="syn20968331")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--token-file", type=Path, default=Path("~/.synapse_token"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    token = args.token_file.expanduser().read_text().strip()
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    files = synapseutils.syncFromSynapse(syn, args.entity, path=str(args.out), ifcollision="keep.local")
    rows = []
    for f in files:
        p = Path(f.path)
        rows.append((p.stat().st_size if p.exists() else -1, str(p.relative_to(args.out)) if p.exists() else p.name))
    rows.sort(key=lambda r: r[1])
    with (args.out / "MANIFEST.txt").open("w") as fh:
        for size, name in rows:
            line = f"{size / 1e6:10.1f} MB  {name}"
            print(line)
            fh.write(line + "\n")
    print(f"{len(rows)} files, {sum(max(s, 0) for s, _ in rows) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
