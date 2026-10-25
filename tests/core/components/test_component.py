"""Test the base of every component, its constructor, and the estimator contract."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import ArgumentError, state
from skcomplex.core.components import Aggregate, BaseComponent, ElementComponent, GroupComponent, Wiener
from skcomplex.core.interactions import BaseInteraction
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.models.biology.ecology import PopulationComponent, PredationInteraction


class PendulumComponent(ElementComponent):
    """Pendulum whose angle has order 2.

    Args:
        label:
            The label of the pendulum.

        theta:
            The initial angle.
    """

    theta: float = state(0.0, order=2)


def test_component_is_the_base_of_every_component():
    """Test that elements, groups, interactions, and systems derive from the component base."""
    for kind in (ElementComponent, GroupComponent, BaseInteraction, System):
        assert issubclass(kind, BaseComponent)
    assert isinstance(System(components=[PopulationComponent('rabbits')]), BaseComponent)


def test_component_stores_companions_without_class_attributes():
    """Test that a state of order 2 takes its derivative under a derived name that is not a class attribute."""
    bob = PendulumComponent('bob', theta=0.1, theta_d1=0.5)
    assert not hasattr(PendulumComponent, 'theta_d1')
    assert (bob.theta, bob.theta_d1) == (0.1, 0.5)
    assert {'theta', 'theta_d1'} <= set(bob.get_params())
    assert list(PendulumComponent.declarations()) == ['theta', 'theta_d1']


def test_component_rejects_unknown_and_missing_arguments():
    """Test that an undeclared value and a missing role raise a named error that is also a type error."""
    with pytest.raises(ArgumentError, match="'foo'"):
        PopulationComponent('rabbits', foo=1)
    growth = BaseInteraction.define('GrowthInteraction', roles=['target'])
    with pytest.raises(TypeError, match="'target'"):
        growth('growth')


def test_component_parameters_list_settings_and_declared_names():
    """Test that the parameters are the label, the settings, and every declared name with its value or default."""
    aggregate = Aggregate('size', group='farmers')
    assert aggregate.get_params() == {
        'group': 'farmers',
        'label': 'size',
        'reduce': 'count',
        'value': 0.0,
        'variable': None,
    }
    assert repr(aggregate) == "Aggregate(group='farmers', label='size')"
    assert Wiener.get_argument_names() == ['dimension', 'covariance']


def test_clone_keeps_declared_values():
    """Test that a clone of a group of 100 members has the same values and compiles to the same problem."""
    group = PopulationComponent.group('herds', 100, n=np.linspace(1.0, 2.0, 100))
    cloned = clone(group)
    assert (cloned.label, cloned.size) == ('herds', 100)
    assert np.array_equal(cloned.values['n'], group.values['n'])
    first, second = compile_system(System(components=[group])), compile_system(System(components=[cloned]))
    assert np.array_equal(first.initial_state, second.initial_state)
    assert first.equations == second.equations
    single = PendulumComponent('bob', theta=0.3)
    assert clone(single).get_params() == single.get_params()


def test_component_supports_estimator_contract():
    """Test that components follow the scikit-learn parameter contract."""
    interaction = PredationInteraction('hunt', predator='foxes', prey='rabbits', a=0.4)
    assert interaction.get_params()['a'] == 0.4
    cloned = clone(interaction)
    assert cloned is not interaction
    assert cloned.a == 0.4
