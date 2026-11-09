"""Test the `_nonnegative` module of `skcomplex.systems.general.elements`."""

from types import SimpleNamespace

import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.general.elements import NonnegativeElement


def test_nonnegative_lists_its_value_as_a_parameter():
    """Test that a nonnegative element lists its label, its visual, and its value `x`."""
    assert NonnegativeElement('S', x=990).get_params() == {'label': 'S', 'visual': None, 'x': 990}


def test_nonnegative_state_has_a_nonnegative_domain():
    """Test that the one state `x` takes values in `[0, oo)`."""
    states = NonnegativeElement('S').get_states()
    assert list(states) == ['x']
    assert states['x'].domain == RealSet.closed_open(0, oo)


def test_nonnegative_keeps_its_value_without_laws():
    """Test that declared elements keep their amounts and fractions, since the element has no laws."""
    system = CompositeSystem(elements=[NonnegativeElement('S', x=990), NonnegativeElement('I', x=0.25)])
    trajectory = simulate(system, until=3, dt=1.0)
    assert trajectory.get_values('S.x').tolist() == [990.0] * 4
    assert trajectory.get_values('I.x').tolist() == [0.25] * 4


def test_nonnegative_runs_into_no_laws():
    """Test that a nonnegative element contributes no laws."""
    assert NonnegativeElement('S', x=1.0).prepare().run(SimpleNamespace(x=var('x'))).laws_ == []


def test_nonnegative_rejects_a_negative_value():
    """Test that a run rejects an element whose value is outside the nonnegative numbers."""
    system = CompositeSystem(elements=[NonnegativeElement('S', x=-1.0)])
    with pytest.raises(ValidationError):
        simulate(system, until=1, dt=1.0)


def test_nonnegative_members_form_a_distribution_over_classes():
    """Test that a group of nonnegative elements on a lattice of classes has one member per class."""
    classes = LatticeSpace(5, boundary='fixed')
    group = GroupElement('distribution', element=NonnegativeElement, space=classes, on='sites', values={'x': 0.2})
    assert group.prepare().size_ == 5
