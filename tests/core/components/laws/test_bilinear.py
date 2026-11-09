"""Test the `_bilinear` module of `skcomplex.core.components.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import BilinearLaw
from skcomplex.core.components.spaces import ContinuousTime, LatticeSpace
from skcomplex.core.errors import ValidationError

F, G, H, MU, Q = var('f g h mu q')


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


def test_bilinear_needs_continuous_time_on_class_lattices():
    """Test that a bilinear law runs in continuous time on lattices of classes."""
    requirements = BilinearLaw(F, F, G, 1.0, np.full((2, 2, 2), 0.5), conserved=True)._get_requirements()
    assert (requirements.time, requirements.space) == ((ContinuousTime,), (LatticeSpace,))


def test_conserved_bilinear_runs_into_its_tables():
    """Test that a conserved law runs into its rate and outcome tables as nested lists of numbers."""
    law = BilinearLaw(F, F, G, [[1.0, 2.0], [3.0, 4.0]], np.full((2, 2, 2), 0.5), conserved=True)
    rate, outcomes = law.prepare().run({}).value_
    assert rate == [[1.0, 2.0], [3.0, 4.0]]
    assert np.shape(outcomes) == (2, 2, 2)


def test_conserved_bilinear_evaluates_parameter_tables():
    """Test that a table with a parameter is evaluated at the value of the parameter, which the law reads."""
    law = BilinearLaw(F, F, F, Q, [[[Q, 1 - Q]], [[0.5, 0.5]]], conserved=True)
    rate, outcomes = law.prepare().run({'q': 0.25}).value_
    assert rate == 0.25
    assert outcomes[0][0] == [0.25, 0.75]
    assert law.reads_ == {F, Q}


@pytest.mark.parametrize(
    ('rate', 'outcomes', 'message'),
    [
        (1.0, np.ones((2, 2)), 'must be an array of 3 dimensions'),
        (np.ones(3), np.ones((2, 2, 2)), 'must be an array of 0 or 2 dimensions'),
        (np.ones((3, 2)), np.ones((2, 2, 2)), 'has the shape'),
    ],
)
def test_conserved_bilinear_rejects_tables_of_wrong_shape(rate, outcomes, message):
    """Test that a conserved law rejects tables of the wrong number of dimensions or of disagreeing classes."""
    with pytest.raises(ValidationError, match=message):
        BilinearLaw(F, F, F, rate, outcomes, conserved=True).prepare()


def test_conserved_bilinear_targets_the_first_density_and_the_target():
    """Test that a conserved law targets the first density, and a target other than it."""
    assert BilinearLaw(F, F, G, 1.0, np.full((2, 2, 2), 0.5), conserved=True).prepare().targets_ == (F,)
    assert BilinearLaw(H, F, G, 1.0, np.full((2, 2, 2), 0.5), conserved=True).prepare().targets_ == (F, H)


def test_conserved_bilinear_needs_outcomes():
    """Test that a conserved law without an outcome table asks for the outcome probabilities."""
    with pytest.raises(ValidationError, match='Give the outcome probabilities'):
        BilinearLaw(F, F, G, 1.0, conserved=True).prepare()


def test_only_a_conserved_bilinear_takes_no_second_density():
    """Test that a law without a second density is a conserved linear redistribution."""
    assert BilinearLaw(F, F, None, 1.0, np.ones((2, 1, 2)) / 2, conserved=True).prepare().targets_ == (F,)
    with pytest.raises(ValidationError, match='only a conserved law takes'):
        BilinearLaw(F, F, None, 1.0, np.ones((2, 1, 2)) / 2).prepare()
