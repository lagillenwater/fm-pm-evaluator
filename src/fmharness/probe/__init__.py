"""The fixed downstream probe shared by every model.

One probe runs on top of every adapter's embedding so a model comparison
isolates the encoder. The minimal harness uses ``SimpleProbe`` (per-drug mean
plus one shared ridge slope on a few PCs); set ``n_components=0`` for the
drug-mean baseline.
"""

from fmharness.probe.simple import SimpleProbe

__all__ = ["SimpleProbe"]
