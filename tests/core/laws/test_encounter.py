"""Test the `_encounter` module of `skcomplex.core.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.laws import EncounterLaw
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.times import ContinuousTime
from skcomplex.errors import ValidationError

F, G, Q = var('f g q')


def test_encounter_needs_continuous_time_on_activity_lattices():
    """Test that an encounter runs in continuous time on lattices of activities."""
    requirements = EncounterLaw(F, G, 1.0, np.full((2, 2, 2), 0.5))._get_requirements()
    assert (requirements.time, requirements.space) == ((ContinuousTime,), (LatticeSpace,))


def test_encounter_runs_into_its_tables():
    """Test that an encounter runs into its rate and transition tables as nested lists of numbers."""
    rate, transition = (
        EncounterLaw(F, G, [[1.0, 2.0], [3.0, 4.0]], np.full((2, 2, 2), 0.5)).check().prepare().run({}).value_
    )
    assert rate == [[1.0, 2.0], [3.0, 4.0]]
    assert np.shape(transition) == (2, 2, 2)


def test_encounter_evaluates_parameter_tables():
    """Test that a table with a parameter is evaluated at the value of the parameter, which the law reads."""
    law = EncounterLaw(F, F, Q, [[[Q, 1 - Q]], [[0.5, 0.5]]]).check()
    rate, transition = law.prepare().run({'q': 0.25}).value_
    assert rate == 0.25
    assert transition[0][0] == [0.25, 0.75]
    assert law.reads_ == {F, Q}


@pytest.mark.parametrize(
    ('rate', 'transition', 'message'),
    [
        (1.0, np.ones((2, 2)), 'must be an array of 3 dimensions'),
        (np.ones(3), np.ones((2, 2, 2)), 'must be an array of 0 or 2 dimensions'),
        (np.ones((3, 2)), np.ones((2, 2, 2)), 'has the shape'),
    ],
)
def test_encounter_rejects_tables_of_wrong_shape(rate, transition, message):
    """Test that an encounter rejects tables of the wrong number of dimensions or of disagreeing activities."""
    with pytest.raises(ValidationError, match=message):
        EncounterLaw(F, F, rate, transition).check()


def test_encounter_targets_the_candidate_and_the_test():
    """Test that an encounter targets the candidate, and a test density other than it."""
    h = var('h')
    assert EncounterLaw(F, G, 1.0, np.full((2, 2, 2), 0.5), test=F).check().prepare().targets_ == (F,)
    assert EncounterLaw(F, G, 1.0, np.full((2, 2, 2), 0.5), test=h).check().prepare().targets_ == (F, h)
