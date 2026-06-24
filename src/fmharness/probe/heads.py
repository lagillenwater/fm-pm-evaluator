"""Probe-head registry: build a probe factory by name.

The transfer and per-patient scripts already take a zero-arg ``probe_factory``
(``grouped_cv_predict``, ``transfer_predict``). ``make_head`` returns such a
factory for a named head so a ``--head`` flag swaps the linear ridge for the
nonlinear kernel ridge without touching the harness. Both heads share the
``fit``/``predict_parts`` contract and the same PCA/NMF reduction, so the only
thing that changes is the residual model -- the comparison stays apples-to-apples.

The bilinear model is intentionally absent: it needs drug fingerprints ``g_d`` and
operates on ``[z, g, z(x)g]``, so it does not satisfy the ``fit(emb, drugs, y)``
contract. It contributes its row to the head-invariance table from its own script.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from fmharness.probe.kernel import KernelProbe
from fmharness.probe.simple import SimpleProbe

__all__ = ["HEADS", "make_head"]

HEADS = ("linear", "kernel")


def make_head(
    name: str,
    *,
    n_components: int = 10,
    std_floor: float = 0.0,
    reducer: str = "pca",
    per_drug: bool = True,
) -> Callable[[], SimpleProbe | KernelProbe]:
    """Return a zero-arg factory for the named head (``"linear"`` or ``"kernel"``)."""
    kwargs = dict(
        n_components=n_components,
        std_floor=std_floor,
        reducer=reducer,
        per_drug=per_drug,
    )
    if name == "linear":
        return partial(SimpleProbe, **kwargs)
    if name == "kernel":
        return partial(KernelProbe, **kwargs)
    raise ValueError(f"unknown head {name!r}; choose from {HEADS}")
