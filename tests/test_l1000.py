"""Tests for the shared L1000 builders (the cmapPy gctx path is Alpine-only)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.l1000 import build_generated_deltas, drug_pert_maps, logcpm


def test_logcpm_is_scale_invariant() -> None:
    # two rows with the same relative profile but different depth -> identical log-CPM
    df = pd.DataFrame([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], columns=pd.Index(list("abc")))
    z = logcpm(df)
    assert np.allclose(z.iloc[0].to_numpy(), z.iloc[1].to_numpy())


def test_drug_pert_maps_cid_and_inchikey() -> None:
    # D3 carries a CID (777) whose CID has no matching pert, but its InChIKey
    # prefix does -- so it must still resolve, keyed by its CID. D2's CID matches
    # neither a pert CID nor InChIKey. Drugs are keyed by PubChem CID (string).
    drugs = pd.DataFrame(
        {
            "improve_drug_id": ["D1", "D2", "D3"],
            "pubchem_id": [123, 999999, 777],
            "InChIKey": ["AAAAAAAAAAAAAA-x", "BBB-y", "CCCCCCCCCCCCCC-z"],
        }
    )
    pert = pd.DataFrame(
        {
            "pert_type": ["trt_cp", "trt_cp", "ctl_vehicle"],
            "pubchem_cid": [123, 0, 5],
            "inchi_key_prefix": ["ZZZ", "CCCCCCCCCCCCCC", "QQ"],
            "pert_id": ["BRD-A", "BRD-C", "BRD-V"],
        }
    )
    drug2pert, pert2drug = drug_pert_maps(drugs, pert)
    assert drug2pert["123"] == "BRD-A"  # matched by PubChem CID 123
    assert drug2pert["777"] == "BRD-C"  # matched by 14-char InChIKey prefix
    assert "999999" not in drug2pert  # no CID / InChIKey match
    assert pert2drug["BRD-A"] == "123"


def _write_adata(path: Path, x: list[list[float]], obs: list[str], var: list[str]) -> None:
    a = ad.AnnData(X=np.asarray(x, dtype=np.float32))
    a.obs_names = obs
    a.var_names = var
    a.write_h5ad(path)


def test_build_generated_deltas(tmp_path: Path) -> None:
    genes, orgs = ["A", "B", "C"], ["o1", "o2"]
    base = tmp_path / "baseline.h5ad"
    _write_adata(base, [[10, 20, 30], [40, 50, 60]], orgs, genes)
    gdir = tmp_path / "gen"
    gdir.mkdir()
    _write_adata(gdir / "BRD-1.h5ad", [[12, 18, 33], [44, 48, 66]], orgs, genes)  # -> drug D1
    _write_adata(gdir / "BRD-X.h5ad", [[1, 1, 1], [1, 1, 1]], orgs, genes)  # unmapped

    delta, key = build_generated_deltas(gdir, base, {"BRD-1": "D1"}, use_logcpm=False)
    assert set(delta.columns) == {"A", "B", "C"}
    assert delta.shape == (2, 3)  # only BRD-1's 2 organoids
    assert list(key["drug"].unique()) == ["D1"]  # BRD-X skipped
    assert float(delta.loc[delta.index[0], "A"]) == 2.0  # 12 - 10 for o1
