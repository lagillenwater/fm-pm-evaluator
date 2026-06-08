"""Controls for the drug-response probe.

Two controls bracket every result:

- Negative (permutation). Shuffle the response within each drug. This breaks
  the link between expression and response while keeping each drug's mean and
  its marginal distribution. Re-running the probe on the shuffled labels gives
  the null distribution of the within-drug score for this exact model and
  sample size. A real result has to beat that null.

- Positive (planted signal). Build a synthetic response that depends on a
  known direction in expression, at a set effect size, plus the real per-drug
  means and noise. Re-running the probe checks that it recovers the planted
  signal. This proves the pipeline detects per-sample expression signal when
  it exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def permute_within_drug(
    drug_ids: pd.Series, y: pd.Series, rng: np.random.Generator
) -> np.ndarray:
    """Return ``y`` shuffled among samples within each drug.

    Per-drug means and marginals are preserved exactly; only the assignment of
    response to sample changes, so this is the null for the within-drug score.
    """
    y_arr = np.asarray(y, dtype=np.float64).copy()
    d = np.asarray(drug_ids, dtype=object)
    for drug in np.unique(d):
        idx = np.flatnonzero(d == drug)
        y_arr[idx] = y_arr[rng.permutation(idx)]
    return y_arr


def plant_response(
    drug_ids: pd.Series,
    y: pd.Series,
    embedding: np.ndarray,
    *,
    effect: float,
    rng: np.random.Generator,
    noise_sd: float = 1.0,
    n_components: int = 30,
) -> np.ndarray:
    """Synthetic response with a known expression signal.

    ``y_synth = drug_mean(d) + effect * signal_i + noise``, where signal_i is a
    standardized projection of the sample embedding onto a fixed random
    direction. ``effect`` is the coefficient on the unit-variance signal, so
    larger ``effect`` means a stronger per-sample (within-drug) effect. Per-drug
    means come from the real responses, so the between-drug structure matches.

    The direction lives in the top ``n_components`` principal components of the
    standardized embedding, the same subspace the probe reduces to. Planting in
    the full gene space would put most of the signal in directions the probe
    discards, so a stronger planted effect would not raise the recovered score.
    Planting in-subspace makes the positive control a real recovery test.
    """
    d = np.asarray(drug_ids, dtype=object)
    y_arr = np.asarray(y, dtype=np.float64)
    drug_mean = pd.Series(y_arr).groupby(d).transform("mean").to_numpy()

    emb = np.asarray(embedding, dtype=np.float64)
    k = min(n_components, max(1, emb.shape[0] - 1), emb.shape[1])
    z = PCA(n_components=k).fit_transform(StandardScaler().fit_transform(emb))
    w = rng.standard_normal(k)
    w /= np.linalg.norm(w)
    signal = z @ w
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)

    noise = rng.standard_normal(len(y_arr)) * noise_sd
    return drug_mean + effect * signal + noise


def plant_interaction(
    drug_ids: pd.Series,
    y: pd.Series,
    embedding: np.ndarray,
    *,
    effect: float,
    rng: np.random.Generator,
    noise_sd: float = 1.0,
    n_components: int = 30,
) -> np.ndarray:
    """Synthetic response with a known drug-specific (interaction) signal.

    Each drug gets its own expression direction in the top-``n_components`` PC
    subspace, and the directions are centered across drugs so the signal has no
    per-organoid mean. That makes it a pure organoid x drug interaction: it
    survives the row-centering in ``interaction_rho`` and vanishes from a model
    that can only fit one shared slope. Use this as the positive control for the
    interaction headline; ``plant_response`` plants general sensitivity instead.
    """
    d = np.asarray(drug_ids, dtype=object)
    y_arr = np.asarray(y, dtype=np.float64)
    drug_mean = pd.Series(y_arr).groupby(d).transform("mean").to_numpy()

    emb = np.asarray(embedding, dtype=np.float64)
    k = min(n_components, max(1, emb.shape[0] - 1), emb.shape[1])
    z = PCA(n_components=k).fit_transform(StandardScaler().fit_transform(emb))

    order, inv = np.unique(d, return_inverse=True)
    directions = rng.standard_normal((len(order), k))
    directions -= directions.mean(axis=0, keepdims=True)  # center -> pure interaction
    signal = np.einsum("ij,ij->i", z, directions[inv])  # per-row z_i . w_{drug(i)}
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)

    noise = rng.standard_normal(len(y_arr)) * noise_sd
    return drug_mean + effect * signal + noise
