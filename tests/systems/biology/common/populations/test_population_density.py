"""Test the `_population_density` module of `skcomplex.systems.biology.common.populations`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.populations import PopulationDensityElement
from skcomplex.systems.general.elements import NonnegativeElement
from skcomplex.systems.general.reactions import ReactionInteraction


def test_population_density_is_a_nonnegative_element_with_the_value_x():
    """Test that the realization subclasses the general element and takes its value, size, and space."""
    element = PopulationDensityElement('a', x=0.5)
    assert isinstance(element, NonnegativeElement)
    assert element.get_params() == {'label': 'a', 'x': 0.5, 'size': None, 'space': None}


def test_population_density_on_a_lattice_has_one_value_per_class():
    """Test that the realization on a lattice of classes holds one value per class."""
    density = PopulationDensityElement('density', x=np.full(4, 0.25), space=LatticeSpace(4, boundary='fixed'))
    decay = ReactionInteraction('decay', target='density', reactants={'x': 1}, k=1.0)
    trajectory = System(elements=[density], interactions=[decay]).prepare().simulate(until=1, dt=1.0)
    np.testing.assert_allclose(trajectory.get_lattice('density.x')[-1], np.full(4, 0.25 * np.exp(-1.0)), rtol=1e-4)


def test_population_density_rejects_a_negative_value():
    """Test that preparing rejects a negative value."""
    with pytest.raises(ValidationError):
        System(elements=[PopulationDensityElement('a', x=-1.0)]).prepare()
