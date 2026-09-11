"""The design's model and comparison declarations are complete and self-consistent.

Design.md sections 5 and 7 fix the comparison table exactly: 6 leave-one-line-out (LOLO)
comparisons and 5 leave-one-drug-out (LODO) comparisons, each naming two declared models that
both run under that comparison's scheme.
"""

from __future__ import annotations

from fmharness.heldout import COMPARISONS, MODELS


def test_lolo_and_lodo_comparison_counts() -> None:
    lolo = [c for c in COMPARISONS if c.scheme == "lolo"]
    lodo = [c for c in COMPARISONS if c.scheme == "lodo"]
    assert len(lolo) == 6
    assert len(lodo) == 5


def test_every_comparison_names_declared_models_that_run_on_its_scheme() -> None:
    for comparison in COMPARISONS:
        for model_id in (comparison.model_a, comparison.model_b):
            assert model_id in MODELS, f"{comparison.id} names undeclared model {model_id!r}"
            spec = MODELS[model_id]
            assert comparison.scheme in spec.schemes, (
                f"{comparison.id} runs {model_id!r} under {comparison.scheme!r}, "
                f"but that model only declares {sorted(spec.schemes)!r}"
            )


def test_comparison_ids_are_unique() -> None:
    ids = [c.id for c in COMPARISONS]
    assert len(ids) == len(set(ids))


def test_every_random_model_stands_in_for_a_declared_description() -> None:
    described = {spec.description for spec in MODELS.values() if spec.kind != "random"}
    for spec in MODELS.values():
        if spec.kind == "random":
            assert spec.description in described
