"""Test the `_element` module of `skcomplex.core.components.elements`."""

from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var
from sklearn.base import clone

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import JumpLaw, RateLaw, RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler, compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import LabelError, ValidationError
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class TallyElement(Element):
    """Element with a setting next to its declared state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
        source: str = '',
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.source = source


class DecayingElement(Element):
    """Stock that decays at its own rate."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
        k: Annotated[object, ParameterSpec()] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.k = k

    def _get_laws(self, s):
        """Return the decay of the stock."""
        return [RateLaw(s.n, -s.k * s.n)]


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
        return [RateLaw(s.debt, s.r * s.debt * (1 - _leverage(s)))]


class PendulumElement(Element):
    """Pendulum whose angle has order 2 and takes its derivative as a companion argument."""

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


class CounterElement(Element):
    """Counter whose jump has a parameter as its increment."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
        k: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.k = k

    def _get_laws(self, s):
        """Return the jump of the counter."""
        return [JumpLaw(1.0, {s.n: s.k})]


class MovingElement(Element):
    """Element with a vector velocity whose rate is written with a NumPy scalar factor."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[object, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v

    def _get_laws(self, s):
        """Return the rate of the velocity."""
        return [RateLaw(s.v, np.float64(2.0) * s.v)]


class NumpyDecayElement(Element):
    """Decay and inflow written with NumPy scalars."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the decay and the inflow."""
        return [RateLaw(s.x, np.float64(-0.5) * s.x), RateLaw(s.x, np.float64(1.0))]


class FloatDecayElement(NumpyDecayElement):
    """Decay and inflow written with Python floats."""

    def _get_laws(self, s):
        """Return the decay and the inflow."""
        return [RateLaw(s.x, -0.5 * s.x), RateLaw(s.x, 1.0)]


class SettlerElement(Element):
    """Agent with a state named like an attribute of the agent view."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive

    def _get_laws(self, s):
        """Return a rule that does nothing."""
        return [RuleLaw(self.stay)]

    def stay(self, agent, system):
        """Do nothing."""
        del agent, system


def _leverage(s):
    """Return the leverage of a firm, its debt over its equity."""
    return s.debt / s.equity


def _firm_leverage(s):
    """Return the leverage of the firm of a system."""
    return _leverage(s.firm)


def test_element_has_no_laws_by_default():
    """Test that an element that returns no laws of its own runs into none."""
    assert TallyElement('t').prepare().run(SimpleNamespace()).laws_ == []


def test_element_runs_into_its_laws():
    """Test that an element runs into the laws it writes for a namespace of symbols."""
    n, k = var('n k')
    (law,) = DecayingElement('a').prepare().run(SimpleNamespace(n=n, k=k)).laws_
    assert isinstance(law, RateLaw)
    assert law.value == -k * n


def test_element_parameters_list_settings_and_declared_names():
    """Test that the parameters are the label, the settings, the visual, and every state with its value."""
    tally = TallyElement('size', source='farmers')
    assert tally.get_params() == {'label': 'size', 'source': 'farmers', 'value': 0.0, 'visual': None}
    assert repr(tally) == "TallyElement(label='size', source='farmers')"


def test_element_rejects_unknown_arguments():
    """Test that an element cannot be made with an argument it does not declare."""
    with pytest.raises(TypeError, match="'foo'"):
        TallyElement('rabbits', foo=1)


def test_element_stores_companions_as_arguments():
    """Test that a state of order 2 takes its derivative as an argument of its own, not a class attribute."""
    bob = PendulumElement('bob', theta=0.1, theta_d1=0.5)
    assert not hasattr(PendulumElement, 'theta_d1')
    assert (bob.theta, bob.theta_d1) == (0.1, 0.5)
    assert list(PendulumElement._get_specs()) == ['theta']
    assert clone(bob).get_params() == bob.get_params()


def test_element_clones_and_pickles_with_its_values():
    """Test that an element clones and pickles with its values."""
    restored = ForkingPickler.loads(ForkingPickler.dumps(DecayingElement('a', n=3.0)))
    assert (restored.n, restored.k) == (3.0, 0.2)
    assert clone(DecayingElement('a', n=3.0)).n == 3.0


def test_validation_reports_an_element_with_an_invalid_label_by_its_path():
    """Test that validating a system reports a nested element whose label is not a string."""
    system = System(systems=[System('meadow', elements=[TallyElement(1)])])
    report = PassCompiler().prepare(system).report_
    (issue,) = report.errors
    assert issue.code == 'label'
    assert issue.message.startswith('Label 1 in meadow/ is not valid')


def test_element_decays_at_its_own_rate():
    """Test that the laws of an element run with its own parameters."""
    trajectory = Simulation(System(elements=[DecayingElement('a', n=1.0)]), dt=1).prepare().run(until=1).trajectory_
    assert np.isclose(trajectory.values('a.n')[-1], np.exp(-0.2), rtol=1e-3)


def test_expression_of_states_is_an_observable_of_the_laws():
    """Test that a law computes the leverage as an expression and the run records an observable of the same one."""
    system = System(elements=[FirmElement('firm')])
    assert str(compile_system(system).equations['firm.debt']) == '-debt_firm*r_firm*(debt_firm/equity_firm - 1)'
    leverage = ObservableSpec('leverage', _firm_leverage)
    trajectory = Simulation(system, dt=0.5, observables=[leverage]).prepare().run(until=2).trajectory_
    debt, equity = trajectory.values('firm.debt'), trajectory.values('firm.equity')
    np.testing.assert_allclose(trajectory.values('leverage'), debt / equity)


def test_agent_state_named_like_view_raises_error():
    """Test that an agent with a state named like an attribute of the agent view raises a label error."""
    with pytest.raises(LabelError, match=r'settlers declare the variables alive,.*reserved'):
        compile_system(System(elements=[GroupElement('settlers', element=SettlerElement, size=2)]))


def test_numpy_scalar_times_vector_is_rejected():
    """Test that a law value that NumPy built as an array raises a named error that names the state."""
    system = System(elements=[MovingElement('m', v=[1.0, 2.0])], space=ContinuousSpace(n_dim=2))
    with pytest.raises(ValidationError, match='NumPy array'):
        compile_system(system)


def test_numpy_scalar_values_match_floats():
    """Test that NumPy scalar law values run like the same laws written with Python floats."""
    first, second = (
        Simulation(System(elements=[element('d')]), dt=0.5).prepare().run(until=2).trajectory_.values('d.x')
        for element in (NumpyDecayElement, FloatDecayElement)
    )
    assert np.array_equal(first, second)


def test_jump_increment_symbol_is_rejected():
    """Test that a jump whose increment is a parameter raises a named error naming the state and the increment."""
    with pytest.raises(ValidationError, match=r'jump of n_c has the increment k_c.*number'):
        compile_system(System(elements=[CounterElement('c')]))
