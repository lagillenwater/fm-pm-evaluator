"""Tests for the shared L1000 builders (the cmapPy gctx path is Alpine-only)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from fmharness.l1000 import (
    build_additive_deltas,
    build_generated_deltas,
    build_learned_deltas,
    drug_pert_maps,
    logcpm,
)


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


def test_build_additive_deltas_is_drug_mean_per_organoid() -> None:
    # two drugs over cell lines L1/L2; the additive delta is each drug's mean over its
    # lines, assigned identically to every organoid (no organoid x drug interaction).
    genes = ["A", "B"]
    l1000_delta = pd.DataFrame(
        [[2.0, 4.0], [4.0, 8.0], [1.0, 1.0], [3.0, 3.0]], columns=pd.Index(genes)
    )
    l1000_key = pd.DataFrame(
        {"patient": ["L1", "L2", "L1", "L2"], "drug": ["d1", "d1", "d2", "d2"]}
    )
    delta, key = build_additive_deltas(l1000_delta, l1000_key, ["o1", "o2", "o3"])

    assert list(delta.columns) == genes
    assert delta.shape == (2 * 3, 2)  # 2 drugs x 3 organoids
    # every organoid gets d1's mean delta [3, 6] and d2's mean delta [2, 2]
    for drug, want in (("d1", [3.0, 6.0]), ("d2", [2.0, 2.0])):
        rows = delta[key["drug"].to_numpy() == drug].to_numpy()
        assert rows.shape == (3, 2)
        assert np.allclose(rows, want)  # organoid-independent
    assert set(key["patient"]) == {"o1", "o2", "o3"}


def test_build_learned_deltas_is_drug_mean_plus_organoid_correction() -> None:
    # learned predictor: delta(organoid, drug) = drug_mean[drug] + correction(organoid).
    # The correction is drug-independent, so within an organoid the difference between
    # two drugs' predicted deltas equals the difference of their drug means -- exactly,
    # regardless of the fitted ridge. And different organoids get different deltas.
    genes = pd.Index(["A", "B", "C", "D"])
    rng = np.random.default_rng(0)
    cells = [f"L{i}" for i in range(6)]
    train_base = pd.DataFrame(rng.random((6, 4)) + 0.5, index=pd.Index(cells), columns=genes)
    keys = [(c, d) for d in ("d1", "d2") for c in cells]
    train_key = pd.DataFrame(keys, columns=pd.Index(["patient", "drug"]))
    dmean = {"d1": np.array([1.0, 2.0, 3.0, 4.0]), "d2": np.array([-1.0, 0.0, 1.0, 2.0])}
    base_arr = train_base.loc[[p for p, _ in keys]].to_numpy()
    delta_rows = [dmean[d] + 0.1 * base_arr[i] for i, (_, d) in enumerate(keys)]
    train_delta = pd.DataFrame(np.asarray(delta_rows), columns=genes)
    target_base = pd.DataFrame(
        rng.random((3, 4)) + 0.5, index=pd.Index(["o1", "o2", "o3"]), columns=genes
    )

    delta, key = build_learned_deltas(
        train_base, train_delta, train_key, target_base, ["o1", "o2", "o3"], reducer="pca", k=3
    )
    assert delta.shape == (2 * 3, 4)  # 2 drugs x 3 organoids
    assert list(delta.columns) == list(genes)

    want_diff = dmean["d1"] - dmean["d2"]
    for p in ("o1", "o2", "o3"):
        d1 = delta[(key["patient"] == p) & (key["drug"] == "d1")].to_numpy()[0]
        d2 = delta[(key["patient"] == p) & (key["drug"] == "d2")].to_numpy()[0]
        assert np.allclose(d1 - d2, want_diff)  # correction cancels -> drug-mean difference
    # organoid-specific: o1 and o2 do not get identical predicted deltas
    o1 = delta[(key["patient"] == "o1") & (key["drug"] == "d1")].to_numpy()[0]
    o2 = delta[(key["patient"] == "o2") & (key["drug"] == "d1")].to_numpy()[0]
    assert not np.allclose(o1, o2)
