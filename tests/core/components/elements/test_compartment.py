"""Test the `_compartment` module of `skcomplex.core.components.elements`."""

from types import SimpleNamespace

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError


def test_compartment_lists_its_amount_as_a_parameter():
    """Test that a compartment lists its label, its visual, and its amount `n`."""
    assert CompartmentElement('S', n=990).get_params() == {'label': 'S', 'n': 990, 'visual': None}


def test_compartment_keeps_its_amount_without_laws():
    """Test that declared compartments keep their counts and fractions, since a compartment has no laws."""
    system = System(elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=0.25)])
    trajectory = simulate(system, until=3, dt=1.0)
    assert trajectory.values('S.n').tolist() == [990.0] * 4
    assert trajectory.values('I.n').tolist() == [0.25] * 4


def test_compartment_runs_into_no_laws():
    """Test that a compartment contributes no laws."""
    assert CompartmentElement('S', n=1.0).prepare().run(SimpleNamespace(n=var('n'))).laws_ == []


def test_compartment_rejects_a_negative_amount():
    """Test that a run rejects a compartment whose amount is outside the nonnegative numbers."""
    system = System(elements=[CompartmentElement('S', n=-1.0)])
    with pytest.raises(ValidationError):
        simulate(system, until=1, dt=1.0)
