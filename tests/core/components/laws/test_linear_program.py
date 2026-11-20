"""Test the `_linear_program` module of `skcomplex.core.components.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import LinearObjectiveSpec
from skcomplex.core.components.laws._linear_program import _LinearProgramLaw
from skcomplex.core.components.spaces._continuous_time import _ContinuousTime
from skcomplex.core.components.spaces._discrete_time import _DiscreteTime
from skcomplex.core.components.spaces._static_time import _StaticTime
from skcomplex.core.errors import ValidationError

V, W = var('v w')


def test_get_requirements_takes_continuous_discrete_or_static_time():
    """Test that the law requires continuous, discrete, or static time."""
    requirements = _LinearProgramLaw([V], [V <= 1])._get_requirements()
    assert set(requirements.time) == {_ContinuousTime, _DiscreteTime, _StaticTime}


def test_prepare_targets_the_variables():
    """Test that the targets of the law are its variables."""
    assert _LinearProgramLaw([V, W], [V + W <= 1]).prepare('flows').targets_ == (V, W)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        (([], []), 'at least one variable'),
        (([V], [V + 1]), 'Sage relation'),
        (([V], [V < 1]), 'Sage relation'),
        (([V], [V <= 1], [V]), 'LinearObjectiveSpec'),
    ],
)
def test_check_rejects_a_malformed_program(arguments, message):
    """Test that a program without variables, a malformed constraint, or a malformed objective is rejected."""
    with pytest.raises(ValidationError, match=message):
        _LinearProgramLaw(*arguments).prepare()


def test_check_rejects_an_unknown_policy_and_sense():
    """Test that an unknown infeasibility policy and an unknown sense of an objective are rejected."""
    with pytest.raises(ValidationError, match='raise, zero, hold'):
        _LinearProgramLaw([V], [V <= 1], infeasible='stop').prepare()
    with pytest.raises(ValidationError, match='max, min'):
        _LinearProgramLaw([V], [V <= 1], [LinearObjectiveSpec(V, sense='most')]).prepare()
