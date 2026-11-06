"""Test the `_bilinear` module of `skcomplex.core.components.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import BilinearLaw
from skcomplex.core.errors import ValidationError

F, G, H, MU = var('f g h mu')


def test_bilinear_runs_into_its_signed_rate_and_outcomes():
    """Test that a bilinear law runs into its signed rate and its outcomes at the values of its parameters."""
    assert BilinearLaw(F, F, G, 0.5).prepare().run({}).value_ == (0.5, None)
    assert BilinearLaw(F, F, G, -2 * MU).prepare().run({'mu': 0.1}).value_[0] == pytest.approx(-0.2)
    rate, outcomes = BilinearLaw(G, F, G, 0.5, np.ones((1, 1, 1))).prepare().run({}).value_
    assert (rate, outcomes) == (0.5, [[[1.0]]])


def test_bilinear_targets_the_first_density_and_the_target():
    """Test that a bilinear law targets the first density, and a target other than it."""
    assert BilinearLaw(F, F, G, 0.5).prepare().targets_ == (F,)
    assert BilinearLaw(H, F, G, 0.5, np.ones((1, 1, 1))).prepare().targets_ == (F, H)


def test_bilinear_rejects_tables_of_wrong_shape():
    """Test that a rate or outcomes of the wrong dimensions are rejected."""
    with pytest.raises(ValidationError, match='0 or 2 dimensions'):
        BilinearLaw(F, F, G, np.ones(3)).prepare()
    with pytest.raises(ValidationError, match='3 dimensions'):
        BilinearLaw(F, F, G, 0.5, np.ones((2, 2))).prepare()
