"""Test the description layer: components, laws, symbols, spaces, and systems."""

import pytest
from sklearn.base import clone

from skcomplex.core.model import EuclideanSpace, Symbols, System
from skcomplex.models.biology.ecology import Population, Predation


def test_element_stores_arguments():
    """Test that an element stores its label and initial state unmodified."""
    element = Population('rabbits', n=40)
    assert element.label == 'rabbits'
    assert element.n == 40


def test_interaction_requires_roles():
    """Test that an interaction without its roles cannot be created."""
    with pytest.raises(TypeError, match='predator'):
        Predation('hunt', prey='rabbits')


def test_component_supports_estimator_contract():
    """Test that components follow the scikit-learn parameter contract."""
    interaction = Predation('hunt', predator='foxes', prey='rabbits', a=0.4)
    assert interaction.get_params()['a'] == 0.4
    cloned = clone(interaction)
    assert cloned is not interaction
    assert cloned.a == 0.4


def test_symbols_unknown_name_lists_available_names():
    """Test that a missing symbol names the available ones."""
    with pytest.raises(AttributeError, match='Available names are: a'):
        _ = Symbols(a=1).b


def test_system_repr_counts_children():
    """Test that a system shows its label and the number of each child kind."""
    system = System('meadow', elements=[Population('rabbits')], space=EuclideanSpace(2))
    assert repr(system) == "System('meadow', elements=1, interactions=0, systems=0)"
