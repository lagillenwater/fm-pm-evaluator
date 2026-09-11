"""Morgan fingerprints and Tanimoto similarity for rung 1's drug descriptions (task 5).

Design.md section 4: a drug hidden from the model (leave-one-drug-out) is described by the
small chemical groups it contains -- a Morgan fingerprint computed from its SMILES string --
and two drugs' similarity is the share of groups they have in common (Tanimoto similarity).
This module holds both pure, testable pieces; the SMILES themselves come from Tahoe's drug
metadata table via ``fmharness.heldout.grid.load_drug_metadata``.
"""

# rdkit ships no PEP-561 type stubs in this environment; under strict mode that turns every
# call site into a cascade of reportUnknown* noise about *rdkit's* types, not ours (and, since
# its stub-less Mol type is inferred as never-None, a false reportUnnecessaryComparison on the
# very "did parsing fail" check this module exists to make). Same suppression, same rationale
# as the rest of this project's pyright strict config where it touches scientific packages --
# the rules that check our own code stay on.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnnecessaryComparison=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def morgan_fingerprints(smiles: Sequence[str], radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """A bool ``(D, n_bits)`` Morgan fingerprint per SMILES string, in the given order.

    Uses ``rdFingerprintGenerator.GetMorganGenerator`` (the non-deprecated API). Raises
    ``ValueError`` naming every SMILES string RDKit cannot parse, so a single bad entry never
    silently drops a drug or shifts the remaining rows out of alignment.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    unparseable: list[str] = []
    rows: list[np.ndarray] = []
    for s in smiles:
        mol = cast(Any, Chem.MolFromSmiles(s))
        if mol is None:
            unparseable.append(s)
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=bool)
        on_bits = np.array(fp.GetOnBits(), dtype=np.int64)
        if on_bits.size:
            arr[on_bits] = True
        rows.append(arr)
    if unparseable:
        raise ValueError(
            f"RDKit could not parse {len(unparseable)} SMILES string(s): {unparseable}"
        )
    return np.stack(rows, axis=0)


def tanimoto(fingerprints: np.ndarray) -> np.ndarray:
    """The ``(D, D)`` Tanimoto similarity matrix of a bool ``(D, n_bits)`` fingerprint array.

    Vectorized: ``inter = F @ F.T`` (as integers), ``union = a_i + a_j - inter``; similarity is
    ``inter / union``, 0 where the union is 0 (both fingerprints have no bits set at all). The
    diagonal is 1 wherever a fingerprint has any bit set (a fingerprint is always identical to
    itself; a fingerprint with no bits set gives an empty union against itself, so its
    self-similarity comes out 0 -- there is nothing to be identical about).
    """
    f_int = fingerprints.astype(np.int64)
    inter = f_int @ f_int.T
    counts = f_int.sum(axis=1)
    union = counts[:, None] + counts[None, :] - inter
    sim = np.zeros(union.shape, dtype=np.float64)
    nonzero = union > 0
    sim[nonzero] = inter[nonzero] / union[nonzero]
    return sim


__all__ = ["morgan_fingerprints", "tanimoto"]
