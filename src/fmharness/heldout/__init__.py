"""Rung 1's model and comparison declarations (design.md sections 5 and 7).

``MODELS`` names every model rung 1 fits, what it is (a fixed reference, a ridge regression
over a line description, nearest-lines, or a random stand-in), and which held-out scheme(s) it
runs under. ``COMPARISONS`` names every head-to-head test read against a hypothesis: which two
models, on which scheme, and in which direction (``model_a`` minus ``model_b``).

Declared once here so later tasks (fitting, scoring, comparing) import the same ids rather
than re-typing them, and so a single test can check the design table is complete and
internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelKind = Literal["reference", "ridge", "nearest_lines", "random"]
Scheme = Literal["lolo", "lodo"]
Description = Literal[
    "expression", "pca", "nmf", "stack_base", "stack_cytokine", "stack_drug"
]
Hypothesis = Literal["H1(a)", "H1(b)", "H2"]


@dataclass(frozen=True)
class ModelSpec:
    """One model rung 1 fits and scores.

    ``description`` names the line description a model is built from (one of the six
    ``Description`` values), or ``None`` for a model that uses no line description (the two
    fixed references).
    """

    id: str
    kind: ModelKind
    schemes: frozenset[Scheme]
    description: Description | None


@dataclass(frozen=True)
class Comparison:
    """One head-to-head test: ``model_a`` minus ``model_b``, on ``scheme``, for ``hypothesis``."""

    id: str
    scheme: Scheme
    model_a: str
    model_b: str
    hypothesis: Hypothesis


_LOLO_LODO: frozenset[Scheme] = frozenset({"lolo", "lodo"})

#: The design's nine named models plus a random stand-in for each of the six descriptions.
MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec("drug_average", "reference", frozenset({"lolo"}), None),
        ModelSpec("chemistry_only", "reference", frozenset({"lodo"}), None),
        ModelSpec("expression", "ridge", _LOLO_LODO, "expression"),
        ModelSpec("pca", "ridge", _LOLO_LODO, "pca"),
        ModelSpec("nmf", "ridge", _LOLO_LODO, "nmf"),
        ModelSpec("stack_base", "ridge", _LOLO_LODO, "stack_base"),
        ModelSpec("stack_cytokine", "ridge", _LOLO_LODO, "stack_cytokine"),
        ModelSpec("stack_drug", "ridge", _LOLO_LODO, "stack_drug"),
        ModelSpec("nearest_lines", "nearest_lines", frozenset({"lolo"}), "expression"),
        ModelSpec("random_expression", "random", _LOLO_LODO, "expression"),
        ModelSpec("random_pca", "random", _LOLO_LODO, "pca"),
        ModelSpec("random_nmf", "random", _LOLO_LODO, "nmf"),
        ModelSpec("random_stack_base", "random", _LOLO_LODO, "stack_base"),
        ModelSpec("random_stack_cytokine", "random", _LOLO_LODO, "stack_cytokine"),
        ModelSpec("random_stack_drug", "random", _LOLO_LODO, "stack_drug"),
    )
}


def _comparison(scheme: Scheme, model_a: str, model_b: str, hypothesis: Hypothesis) -> Comparison:
    return Comparison(f"{scheme}_{model_a}_vs_{model_b}", scheme, model_a, model_b, hypothesis)


#: Design section 7's table: 6 leave-one-line-out comparisons, then 5 leave-one-drug-out.
COMPARISONS: tuple[Comparison, ...] = (
    _comparison("lolo", "stack_base", "drug_average", "H1(a)"),
    _comparison("lolo", "stack_base", "expression", "H1(b)"),
    _comparison("lolo", "stack_base", "pca", "H1(b)"),
    _comparison("lolo", "stack_base", "nmf", "H1(b)"),
    _comparison("lolo", "stack_base", "nearest_lines", "H1(b)"),
    _comparison("lolo", "stack_drug", "stack_cytokine", "H2"),
    _comparison("lodo", "stack_base", "chemistry_only", "H1(a)"),
    _comparison("lodo", "stack_base", "expression", "H1(b)"),
    _comparison("lodo", "stack_base", "pca", "H1(b)"),
    _comparison("lodo", "stack_base", "nmf", "H1(b)"),
    _comparison("lodo", "stack_drug", "stack_cytokine", "H2"),
)

__all__ = ["Comparison", "ModelSpec", "MODELS", "COMPARISONS"]
