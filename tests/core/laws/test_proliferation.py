"""Test the `_proliferation` module of `skcomplex.core.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.laws import ProliferationLaw
from skcomplex.errors import ValidationError

F, G, H = var('f g h')


def test_proliferation_keeps_its_rate_and_outcomes():
    """Test that a proliferation runs into its rate and no outcomes for the candidate's own activity."""
    assert ProliferationLaw(F, G, 0.5).check().prepare().run({}).value_ == (0.5, None)
    rate, outcomes = ProliferationLaw(F, G, 0.5, np.ones((1, 1, 1)), test=G).check().prepare().run({}).value_
    assert (rate, outcomes) == (0.5, [[[1.0]]])


def test_proliferation_targets_the_candidate_and_the_test():
    """Test that a proliferation targets the candidate, and a test density other than it."""
    assert ProliferationLaw(F, G, 0.5).check().prepare().targets_ == (F,)
    assert ProliferationLaw(F, G, 0.5, np.ones((1, 1, 1)), test=H).check().prepare().targets_ == (F, H)


def test_proliferation_rejects_outcomes_of_wrong_shape():
    """Test that outcomes that are not an array of three dimensions are rejected."""
    with pytest.raises(ValidationError, match='3 dimensions'):
        ProliferationLaw(F, G, 0.5, np.ones((2, 2))).check()
