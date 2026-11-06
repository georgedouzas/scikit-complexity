"""Test the `_population` module of `skcomplex.systems.biology.common.populations`."""

from skcomplex.core.components.elements import CompartmentElement
from skcomplex.systems.biology.common.populations import PopulationElement


def test_population_is_a_compartment_of_size_one_by_default():
    """Test that a population is a compartment whose size starts at 1."""
    population = PopulationElement('rabbits')
    assert isinstance(population, CompartmentElement)
    assert population.n == 1.0
