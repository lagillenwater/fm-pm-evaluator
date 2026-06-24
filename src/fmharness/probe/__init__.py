"""The fixed downstream probe shared by every model.

One probe runs on top of every adapter's embedding so a model comparison
isolates the encoder. ``SimpleProbe`` is the linear head (per-drug mean plus a
ridge slope on a few PCs; ``n_components=0`` gives the drug-mean baseline);
``KernelProbe`` is the nonlinear RBF kernel-ridge head. Both share the per-drug
mean and PCA/NMF reduction from ``probe.base`` and expose the same
``fit``/``predict_parts`` contract, so a representation can be scored under either
to test whether a finding is head-invariant. ``heads.make_head`` builds a probe
factory by name.
"""

from fmharness.probe.heads import make_head
from fmharness.probe.kernel import KernelProbe
from fmharness.probe.simple import SimpleProbe

__all__ = ["KernelProbe", "SimpleProbe", "make_head"]
