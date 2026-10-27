"""Test the `_elements` module of `skcomplex.core.base`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import ArgumentError, symbol
from skcomplex.core.elements import BaseElement, Element, GroupElement
from skcomplex.core.laws import RateLaw
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system, validate
from skcomplex.engine.runtime import Simulation
from skcomplex.models.biology.ecology import PopulationElement, PredationInteraction


class TallyElement(Element):
    """Element with a setting next to its declared state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
        source='',
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.source = source


def _leverage(s):
    """Return the leverage of a firm, its debt over its equity."""
    return s.debt / s.equity


def _firm_leverage(s):
    """Return the leverage of the firm of a system."""
    return _leverage(s.firm)


class FirmElement(Element):
    """Firm whose debt grows at a rate that its leverage, an expression of its states, slows down."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        debt: Annotated[float, StateSpec()] = 1.0,
        equity: Annotated[float, StateSpec()] = 2.0,
        r: Annotated[float, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.debt = debt
        self.equity = equity
        self.r = r

    def _get_laws(self, s):
        """Return the growth of the debt, slowed by the leverage."""
        leverage = _leverage(s)
        return [RateLaw(s.debt, s.r * s.debt * (1 - leverage))]


class PendulumElement(Element):
    """Pendulum whose angle has order 2.

    Args:
        label:
            The label of the pendulum.

        theta:
            The initial angle.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(order=2)] = 0.0,
        theta_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1


def test_component_is_the_base_of_elements_and_groups_only():
    """Test that elements and groups derive from the component base, and interactions and systems do not."""
    for kind in (Element, GroupElement):
        assert issubclass(kind, BaseElement)
    for kind in (BaseInteraction, System):
        assert not issubclass(kind, BaseElement)
    system = System(systems=[System('meadow', elements=[PopulationElement('rabbits')])])
    assert not isinstance(system, BaseElement)


def test_component_stores_companions_without_class_attributes():
    """Test that a state of order 2 takes its derivative under a derived name that is not a class attribute."""
    bob = PendulumElement('bob', theta=0.1, theta_d1=0.5)
    assert not hasattr(PendulumElement, 'theta_d1')
    assert (bob.theta, bob.theta_d1) == (0.1, 0.5)
    assert {'theta', 'theta_d1'} <= set(bob.get_params())
    assert list(bob.prepare().declarations_) == ['theta', 'theta_d1']


def test_component_rejects_unknown_and_missing_arguments():
    """Test that an undeclared value and a missing role are stored, and checking raises a named type error."""
    rabbits = PopulationElement('rabbits', foo=1)
    assert rabbits.foo == 1
    with pytest.raises(ArgumentError, match="'foo'"):
        rabbits.check()
    growth = type('GrowthInteraction', (BaseInteraction,), {'target': role()})('growth')
    with pytest.raises(TypeError, match="'target'"):
        growth.check()


def test_validation_checks_every_component_with_its_path():
    """Test that validating a system reports a component with an undeclared value by its path."""
    system = System(systems=[System('meadow', elements=[PopulationElement('rabbits', foo=1)])])
    report = validate(system)
    assert not report.ok
    assert report.errors[0].message.startswith("meadow/rabbits: PopulationElement got 'foo'")


def test_component_runs_its_laws_and_prepares_its_declarations():
    """Test that `run` returns the laws of the `laws` hook and `prepare` sets the declarations."""
    growth = type(
        'GrowthInteraction',
        (BaseInteraction,),
        {'target': role(), 'r': parameter(1.0), 'laws': lambda self, s: [RateLaw(s.target.n, s.r * s.target.n)]},
    )('growth', target='rabbits').prepare()
    assert list(growth.declarations_) == ['target', 'r']
    n, r = symbol('n'), symbol('r')
    (law,) = growth.run(SimpleNamespace(target=SimpleNamespace(n=n), r=r))
    assert isinstance(law, RateLaw)
    assert law.value == r * n


def test_elements_and_interactions_declare_laws_only():
    """Test that elements and interactions declare no quantities, so a system holds no part besides them."""
    for kind in (BaseElement, BaseInteraction, Element, GroupElement):
        assert not hasattr(kind, 'quantities')


def test_auxiliary_is_an_expression_of_the_laws_that_an_observable_records():
    """Test that a law computes the leverage as an expression and the run records an observable of the same one."""
    system = System(elements=[FirmElement('firm')])
    assert str(compile_system(system).equations['firm.debt']) == '-debt_firm*r_firm*(debt_firm/equity_firm - 1)'
    leverage = ObservableSpec('leverage', _firm_leverage)
    trajectory = Simulation(system, dt=0.5, observables=[leverage]).run(until=2).trajectory_
    debt, equity = trajectory.values('firm.debt'), trajectory.values('firm.equity')
    np.testing.assert_allclose(trajectory.values('leverage'), debt / equity)
    assert [type(component) for component in system.components] == [FirmElement]


def test_component_parameters_list_settings_and_declared_names():
    """Test that the parameters are the label, the settings, and every declared name with its value or default."""
    tally = TallyElement('size', source='farmers')
    assert tally.get_params() == {'label': 'size', 'source': 'farmers', 'value': 0.0}
    assert repr(tally) == "TallyElement(label='size', source='farmers', value=0.0)"


def test_clone_keeps_declared_values():
    """Test that a clone of a group of 100 members has the same values and compiles to the same problem."""
    group = GroupElement('herds', element=PopulationElement, size=100, values={'n': np.linspace(1.0, 2.0, 100)})
    cloned = clone(group)
    assert (cloned.label, cloned.size) == ('herds', 100)
    assert np.array_equal(cloned.values['n'], group.values['n'])
    first, second = compile_system(System(elements=[group])), compile_system(System(elements=[cloned]))
    assert np.array_equal(first.initial_state, second.initial_state)
    assert first.equations == second.equations
    single = PendulumElement('bob', theta=0.3)
    assert clone(single).get_params() == single.get_params()


def test_component_supports_estimator_contract():
    """Test that components follow the scikit-learn parameter contract."""
    interaction = PredationInteraction('hunt', predator='foxes', prey='rabbits', a=0.4)
    assert interaction.get_params()['a'] == 0.4
    cloned = clone(interaction)
    assert cloned is not interaction
    assert cloned.a == 0.4
