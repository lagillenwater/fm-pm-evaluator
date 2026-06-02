"""Per-model ``ModelAdapter`` implementations.

One module per model: ``linear_baseline`` (Day 7), with ``stack`` (Day 9) and
``tahoe_x1`` (Day 11) following the same contract.
"""

from fmharness.models.wrappers.linear_baseline import LinearBaselineAdapter

__all__ = ["LinearBaselineAdapter"]
