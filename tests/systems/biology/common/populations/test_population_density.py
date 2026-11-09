"""Test the `_population_density` module of `skcomplex.systems.biology.common.populations`."""

import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import GroupElement, NonnegativeElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.common.populations import PopulationDensityElement


def test_population_density_is_a_nonnegative_element_with_the_state_x():
    """Test that the realization subclasses the core atom and keeps its one state `x` in `[0, oo)`."""
    element = PopulationDensityElement('a', x=0.5)
    assert isinstance(element, NonnegativeElement)
    assert element.get_params() == {'label': 'a', 'visual': None, 'x': 0.5}
    assert element.get_states()['x'].domain == RealSet.closed_open(0, oo)


def test_population_density_group_has_one_member_per_class():
    """Test that a group of the realization on a lattice of classes has one member per class."""
    group = GroupElement(
        'density',
        element=PopulationDensityElement,
        space=LatticeSpace(4, boundary='fixed'),
        on='sites',
        values={'x': 0.25},
    )
    assert group.prepare().size_ == 4


def test_population_density_rejects_a_negative_value():
    """Test that a run rejects a negative value."""
    with pytest.raises(ValidationError):
        simulate(CompositeSystem(elements=[PopulationDensityElement('a', x=-1.0)]), until=1, dt=1.0)
