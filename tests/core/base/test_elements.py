"""Test the `_elements` module of `skcomplex.core.base`."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import (
    ArgumentError,
    BaseComponent,
    BaseInteraction,
    ConservationError,
    parameter,
    state,
    symbol,
)
from skcomplex.core.components import ElementComponent, GroupComponent
from skcomplex.core.laws import (
    AuxiliaryQuantity,
    InvariantQuantity,
    RateLaw,
    Symbols,
)
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system, validate
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.ecology import PopulationComponent, PredationInteraction


class TallyComponent(ElementComponent):
    """Element with a setting next to its declared state."""

    value: float = state(0.0)

    def __init__(self, label, *, source='', **values):
        super().__init__(label, **values)
        self.source = source


class TanksComponent(ElementComponent):
    """Two tanks that drain into each other at a rate an auxiliary names, under the invariant of their total."""

    a: float = state(3.0)
    b: float = state(1.0)
    k: float = parameter(0.5)

    def laws(self, s):
        """Return the rate of the first tank and of the second through the auxiliary."""
        outflow, _ = self.quantities(s)
        return [RateLaw(s.a, -outflow.symbol), RateLaw(s.b, outflow.symbol)]

    def quantities(self, s):
        """Return the outflow of the first tank and the invariant of the total."""
        return [AuxiliaryQuantity('outflow', s.k * s.a), InvariantQuantity('total', s.a + s.b)]


class PendulumComponent(ElementComponent):
    """Pendulum whose angle has order 2.

    Args:
        label:
            The label of the pendulum.

        theta:
            The initial angle.
    """

    theta: float = state(0.0, order=2)


def test_component_is_the_base_of_elements_and_groups_only():
    """Test that elements and groups derive from the component base, and interactions and systems do not."""
    for kind in (ElementComponent, GroupComponent):
        assert issubclass(kind, BaseComponent)
    for kind in (BaseInteraction, System):
        assert not issubclass(kind, BaseComponent)
    system = System(components=[System('meadow', components=[PopulationComponent('rabbits')])])
    assert not isinstance(system, BaseComponent)


def test_component_stores_companions_without_class_attributes():
    """Test that a state of order 2 takes its derivative under a derived name that is not a class attribute."""
    bob = PendulumComponent('bob', theta=0.1, theta_d1=0.5)
    assert not hasattr(PendulumComponent, 'theta_d1')
    assert (bob.theta, bob.theta_d1) == (0.1, 0.5)
    assert {'theta', 'theta_d1'} <= set(bob.get_params())
    assert list(bob.prepare().declarations_) == ['theta', 'theta_d1']


def test_component_rejects_unknown_and_missing_arguments():
    """Test that an undeclared value and a missing role are stored, and checking raises a named type error."""
    rabbits = PopulationComponent('rabbits', foo=1)
    assert rabbits.foo == 1
    with pytest.raises(ArgumentError, match="'foo'"):
        rabbits.check()
    growth = BaseInteraction.define('GrowthInteraction', roles=['target'])('growth')
    with pytest.raises(TypeError, match="'target'"):
        growth.check()


def test_validation_checks_every_component_with_its_path():
    """Test that validating a system reports a component with an undeclared value by its path."""
    system = System(components=[System('meadow', components=[PopulationComponent('rabbits', foo=1)])])
    report = validate(system)
    assert not report.ok
    assert report.errors[0].message.startswith("meadow/rabbits: PopulationComponent got 'foo'")


def test_component_runs_its_laws_and_prepares_its_declarations():
    """Test that `run` returns the laws of the `laws` hook and `prepare` sets the declarations."""
    growth = BaseInteraction.define(
        'GrowthInteraction',
        roles=['target'],
        parameters={'r': 1.0},
        laws=lambda s: [RateLaw(s.target.n, s.r * s.target.n)],
    )('growth', target='rabbits').prepare()
    assert list(growth.declarations_) == ['target', 'r']
    n, r = symbol('n'), symbol('r')
    (law,) = growth.run(Symbols(target=Symbols(n=n), r=r))
    assert isinstance(law, RateLaw)
    assert law.value == r * n


def test_quantities_are_none_by_default():
    """Test that a component and an interaction return no quantities unless their type declares some."""
    s = Symbols(n=symbol('n'))
    assert PopulationComponent('rabbits').quantities(s) == []
    assert BaseInteraction('coupling').quantities(s) == []


def test_quantities_substitute_into_laws_and_hold_every_frame():
    """Test that the auxiliary of a component reaches its rates and its invariant is checked at every frame."""
    system = System(components=[TanksComponent('tanks')])
    problem = compile_system(system)
    assert str(problem.equations['tanks.a']) == '-a_tanks*k_tanks'
    trajectory = Simulation(system, dt=0.5).run(until=2).trajectory_
    np.testing.assert_allclose(trajectory.values('tanks.total'), 4.0)
    np.testing.assert_allclose(trajectory.values('tanks.outflow'), 0.5 * trajectory.values('tanks.a'))


def test_violated_quantity_is_checked_at_every_frame():
    """Test that an invariant the laws break raises the conservation error at the first frame that breaks it."""
    LeakyTanksComponent = ElementComponent.define(
        'LeakyTanksComponent',
        states={'a': 1.0},
        laws=lambda s: [RateLaw(s.a, 1.0)],
        quantities=lambda s: [InvariantQuantity('total', s.a)],
    )
    with pytest.raises(ConservationError, match=r'invariant leaky\.total .* at frame 1'):
        Simulation(System(components=[LeakyTanksComponent('leaky')]), dt=0.5).run(until=2)


def test_component_parameters_list_settings_and_declared_names():
    """Test that the parameters are the label, the settings, and every declared name with its value or default."""
    tally = TallyComponent('size', source='farmers')
    assert tally.get_params() == {'label': 'size', 'source': 'farmers', 'value': 0.0}
    assert repr(tally) == "TallyComponent(label='size', source='farmers', value=0.0)"


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
