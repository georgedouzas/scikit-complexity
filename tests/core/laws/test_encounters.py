"""Test the encounter laws of kinetic models and their declarations."""

import numpy as np
import pytest

from skcomplex.core.base import LawClass, LawKind, SpaceKind, TimeKind, ValidationError, symbol
from skcomplex.core.laws import Destruction, Encounter, Proliferation, outcome_sums


@pytest.mark.parametrize('law', [Encounter, Proliferation, Destruction])
def test_encounter_laws_declare_continuous_encounter_kind(law):
    """Test that every encounter law is of the encounter kind, continuous, and needs continuous time and a lattice."""
    assert law.law_kind is LawKind.ENCOUNTER
    assert law.law_kind.law_class is LawClass.CONTINUOUS
    assert law.requires.time == {TimeKind.CONTINUOUS}
    assert law.requires.space == {SpaceKind.LATTICE}


def test_encounter_keeps_numeric_tables_as_floats():
    """Test that an encounter keeps numeric rates and transitions as float arrays and targets the candidate."""
    f, g = symbol('f'), symbol('g')
    law = Encounter(f, g, [[1.0, 2.0], [3.0, 4.0]], np.full((2, 2, 2), 0.5))
    assert law.rate.dtype == float
    assert law.transition.dtype == float
    assert law.target is f
    assert law.outcomes is law.transition


def test_encounter_keeps_parameter_tables_symbolic():
    """Test that a table with a parameter is kept as Sage expressions."""
    f, q = symbol('f'), symbol('q')
    law = Encounter(f, f, q, [[[q, 1 - q]], [[0.5, 0.5]]])
    assert law.transition.dtype == object
    assert str(law.transition[0, 0, 1]) == '-q + 1'


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
    f = symbol('f')
    with pytest.raises(ValidationError, match=message):
        Encounter(f, f, rate, transition)


def test_proliferation_and_destruction_keep_their_rates():
    """Test that proliferation and destruction keep their rates and target the candidate."""
    f, g = symbol('f'), symbol('g')
    assert Proliferation(f, g, 0.5).outcomes is None
    assert Proliferation(f, g, 0.5, np.ones((1, 1, 1)), test=g).test is g
    assert float(Destruction(f, g, 0.2).rate) == 0.2
    assert Destruction(f, g, 0.2).target is f


def test_outcome_sums_add_over_outcomes():
    """Test that the outcome sums add the probabilities over the last axis."""
    table = np.zeros((2, 2, 3))
    table[0, 0] = [0.2, 0.3, 0.4]
    assert outcome_sums(table)[0, 0] == pytest.approx(0.9)
    assert outcome_sums(table)[1, 1] == 0.0
