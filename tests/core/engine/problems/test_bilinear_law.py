"""Test the `_bilinear_law` module of `skcomplex.core.engine.problems`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.laws.utils import bilinear
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.problems._bilinear_law import BilinearLaw
from skcomplex.core.errors import ValidationError

F, G, H, MU, Q = var('f g h mu q')


def lower(target, form):
    """Return the bilinear law of a rate law with a bilinear form."""
    return BilinearLaw._get_lowered(RateLaw(target, form)).prepare()


def test_lowered_law_runs_into_its_signed_rate_and_outcomes():
    """Test that the lowered law runs into its signed rate and its outcomes at the values of its parameters."""
    assert lower(F, bilinear(0.5, None, F, G)).run({}).value_ == (0.5, None)
    assert lower(F, bilinear(-2 * MU, None, F, G)).run({'mu': 0.1}).value_[0] == pytest.approx(-0.2)
    rate, outcomes = lower(G, bilinear(0.5, np.ones((1, 1, 1)), F, G)).run({}).value_
    assert (rate, outcomes) == (0.5, [[[1.0]]])


def test_lowered_law_targets_the_first_density_and_the_target():
    """Test that the lowered law targets the first density, and a target other than it."""
    assert lower(F, bilinear(0.5, None, F, G)).targets_ == (F,)
    assert lower(H, bilinear(0.5, np.ones((1, 1, 1)), F, G)).targets_ == (F, H)


def test_lowered_law_evaluates_parameter_tables():
    """Test that a table with a parameter is evaluated at the value of the parameter, which the law reads."""
    law = lower(F, bilinear(Q, [[[Q, 1 - Q]], [[0.5, 0.5]]], F, F, conserved=True))
    rate, outcomes = law.run({'q': 0.25}).value_
    assert rate == 0.25
    assert outcomes[0][0] == [0.25, 0.75]
    assert law.reads_ == {F, Q}


def test_lowered_law_needs_continuous_time_on_class_lattices():
    """Test that the lowered law runs in continuous time on lattices of classes."""
    requirements = lower(F, bilinear(1.0, np.full((2, 2, 2), 0.5), F, G, conserved=True))._get_requirements()
    assert [time.time_tag for time in requirements.time] == ['continuous']
    assert requirements.space == (LatticeSpace,)


def test_other_laws_are_kept():
    """Test that a rate without a bilinear form and other laws are kept as they are."""
    rate, noise = RateLaw(F, -F), NoiseLaw(F, 1.0)
    assert BilinearLaw._get_lowered(rate) is rate
    assert BilinearLaw._get_lowered(noise) is noise


def test_bilinear_form_is_the_whole_rate():
    """Test that a bilinear form next to other terms of a rate is rejected."""
    with pytest.raises(ValidationError, match='next to other terms'):
        lower(F, bilinear(1.0, None, F, G) - F)


def test_conserved_form_targets_its_first_density():
    """Test that a conserved form changes its first density, whose loss it takes."""
    with pytest.raises(ValidationError, match='Target the first density'):
        lower(H, bilinear(1.0, np.full((2, 2, 2), 0.5), F, G, conserved=True))
