"""Test the `_transition` module of `skcomplex.core.components.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import TransitionLaw
from skcomplex.core.errors import ValidationError


def test_transition_runs_into_its_square_matrix():
    """Test that a transition law without a source runs into its matrix at given parameters."""
    x, q = var('x'), var('q')
    law = TransitionLaw(x, [[q, 1 - q], [0.4, 0.6]]).prepare()
    assert law.run({'q': 0.1}).value_ == [[0.1, 0.9], [0.4, 0.6]]


def test_transition_with_a_source_takes_a_rectangular_matrix():
    """Test that a transition law with a source takes one row per source value and one column per target value."""
    observed, hidden = var('observed'), var('hidden')
    law = TransitionLaw(observed, [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], source=hidden).prepare()
    assert law.targets_ == (observed,)
    assert law.run({}).value_ == [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]


def test_transition_with_a_source_rejects_a_function_matrix():
    """Test that a transition law with a source asks for a symbolic matrix when given a function."""
    observed, hidden = var('observed'), var('hidden')
    with pytest.raises(ValidationError, match=r'symbolic only.*UpdateLaw'):
        TransitionLaw(observed, lambda t, values: [[1.0]], source=hidden).prepare()
