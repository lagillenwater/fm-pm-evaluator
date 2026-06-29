"""Tests for shared evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmharness.evaluation import regret_norm_at_k


def test_regret_norm_at_k() -> None:
    # y_true / y_pred are AUC-like (lower = better). Patient A's predicted ranking puts
    # its true-best drug first (regret 0); patient B's puts it last (regret 1 at k=1,
    # 0 once k covers all 3). Patients with no spread are skipped.
    preds = pd.DataFrame(
        {
            "patient": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "drug": ["d1", "d2", "d3", "d1", "d2", "d3", "d1", "d2"],
            "y_true": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 5.0, 5.0],  # C flat -> skipped
            "y_pred": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0],
        }
    )
    r = regret_norm_at_k(preds, ks=(1, 3))
    assert np.isclose(r[1], 0.5)  # A: 0, B: (30-10)/(30-10)=1 -> mean 0.5
    assert np.isclose(r[3], 0.0)  # top-3 covers every drug for both A and B
