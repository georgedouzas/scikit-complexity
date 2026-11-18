"""Test the `_dynamical` module of `skcomplex.systems.general.dynamics`."""

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.backends import BaseBackend, NumbaBackend
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System
from skcomplex.systems.general.dynamics import DynamicalSystem
from skcomplex.systems.general.dynamics.types import Equations


class LorenzSystem(DynamicalSystem):
    """The Lorenz flow on `DynamicalSystem`, with its parameters under their own names."""

    def __init__(
        self,
        *,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _get_element_label(self) -> str:
        return 'lorenz'

    def _get_states(self) -> dict[str, object]:
        return {'x': 1.0, 'y': 1.0, 'z': 1.0}

    def _get_parameters(self) -> dict[str, object]:
        return {'sigma': self.sigma, 'rho': self.rho, 'beta': self.beta}

    def _get_equations(self) -> Equations:
        return {
            'x': lambda s: s.target.sigma * (s.target.y - s.target.x),
            'y': lambda s: s.target.x * (s.target.rho - s.target.z) - s.target.y,
            'z': lambda s: s.target.x * s.target.y - s.target.beta * s.target.z,
        }


class LorenzInteraction(Interaction):
    """The Lorenz flow written as one interaction."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context.target, context.t
            self.laws_: list[BaseComponent] = [
                RateLaw(s.x.diff(t), s.sigma * (s.y - s.x)),
                RateLaw(s.y.diff(t), s.x * (s.rho - s.z) - s.y),
                RateLaw(s.z.diff(t), s.x * s.y - s.beta * s.z),
            ]


def henon(order: str) -> DynamicalSystem:
    """Return the Hénon map with its states in the given order."""
    equations = {'x': lambda s: 1 - s.target.a * s.target.x**2 + s.target.y, 'y': lambda s: s.target.b * s.target.x}
    return DynamicalSystem(
        states=dict.fromkeys(order, 0.1),
        equations={name: equations[name] for name in order},
        parameters={'a': 1.4, 'b': 0.3},
        time='discrete',
        element='map',
    )


def test_dynamical_system_decays_exponentially():
    """Test that the rate `-k x` decays to `exp(-k t)` and names the state after the element."""
    system = DynamicalSystem(
        states={'x': 1.0},
        equations={'x': lambda s: -s.target.k * s.target.x},
        parameters={'k': 0.5},
        element='tank',
    )
    trajectory = system.prepare().simulate(until=2.0, dt=0.1)
    assert 'tank.x' in trajectory.names
    assert trajectory.get_values('tank.x')[-1] == pytest.approx(np.exp(-1.0), rel=1e-5)


def test_dynamical_system_rejects_a_state_outside_its_allowed_values():
    """Test that a population allowed only nonnegative sizes rejects a negative one."""
    system = DynamicalSystem(
        states={'n': -1.0},
        equations={'n': lambda s: -s.target.n},
        allowed={'n': RealSet.closed_open(0, oo)},
        element='population',
    )
    with pytest.raises(ValidationError, match='n'):
        system.prepare()
    system.set_params(states={'n': 1.0})
    trajectory = system.prepare().simulate(until=1.0, dt=1.0)
    assert trajectory.get_values('population.n')[-1] == pytest.approx(np.exp(-1.0), rel=1e-4)


def test_dynamical_system_rejects_an_equation_of_an_unknown_state():
    """Test that an equation of an undeclared state names it and the declared states."""
    system = DynamicalSystem(states={'x': 1.0}, equations={'y': lambda s: 0.0})
    with pytest.raises(ValidationError, match=r"'y'.*\['x'\]"):
        system.prepare()


def test_dynamical_system_iterates_the_logistic_map():
    """Test that the logistic map of `r = 2` reaches its fixed point `1/2` and is a discrete update."""
    system = DynamicalSystem(
        states={'x': 0.2},
        equations={'x': lambda s: s.target.r * s.target.x * (1 - s.target.x)},
        parameters={'r': 2.0},
        time='discrete',
    )
    trajectory = system.prepare().simulate(until=30)
    assert trajectory.get_values('state.x')[-1] == pytest.approx(0.5, abs=1e-9)
    tags = system.get_tags()
    assert tags.laws == frozenset({'update'})
    assert tags.time == frozenset({'discrete'})


def test_dynamical_system_tags_an_ornstein_uhlenbeck_process():
    """Test that a rate with a noise amplitude is tagged with a rate, a noise, and randomness."""
    system = DynamicalSystem(states={'x': 1.0}, equations={'x': lambda s: -s.target.x}, noises={'x': 0.3})
    tags = system.get_tags()
    assert tags.laws == frozenset({'rate', 'noise'})
    assert tags.randomness == 'stochastic'


def test_dynamical_system_drives_an_acceleration_with_noise():
    """Test that a noise on the acceleration `x''` shakes the velocity of a damped particle."""
    system = DynamicalSystem(
        states={'x': 0.0},
        equations={"x''": lambda s: -s.target.x.diff(s.t)},
        noises={"x''": 0.5},
        element='bead',
    )
    trajectory = system.prepare().simulate(until=1.0, dt=0.1, random_state=0)
    assert system.form_ == 'stochastic differential equations'
    assert np.ptp(trajectory.get_values("bead.x'")) > 0


def test_dynamical_system_rejects_noise_in_discrete_time():
    """Test that a noise amplitude in discrete time asks for continuous time."""
    system = DynamicalSystem(
        states={'x': 1.0},
        equations={'x': lambda s: s.target.x},
        noises={'x': 0.1},
        time='discrete',
    )
    with pytest.raises(ValidationError, match='continuous time'):
        system.prepare()


def test_dynamical_system_maps_read_the_start_of_the_step():
    """Test that the Hénon map gives the same trajectory in both orders of its states."""
    first = henon('xy').prepare().simulate(until=10)
    second = henon('yx').prepare().simulate(until=10)
    for name in ('map.x', 'map.y'):
        np.testing.assert_array_equal(first.get_values(name), second.get_values(name))


def test_dynamical_system_gives_the_rate_of_a_derivative():
    """Test that the equation of `"x''"` is an acceleration, so `x'' = -x` from rest at 1 swings as `cos t`."""
    system = DynamicalSystem(states={'x': 1.0, "x'": 0.0}, equations={"x''": lambda s: -s.target.x}, element='spring')
    trajectory = system.prepare().simulate(until=3.0, dt=0.5)
    assert system.states_ == ('spring.x', "spring.x'")
    np.testing.assert_allclose(trajectory.get_values('spring.x'), np.cos(trajectory.times), atol=1e-5)


def test_dynamical_system_subclass_takes_its_own_parameters():
    """Test that a subclass lists its own parameters and settings and passes a changed one to the next `prepare`."""
    system = LorenzSystem()
    assert sorted(system.get_params(deep=False)) == ['backend', 'beta', 'label', 'rho', 'sigma', 'solver']
    element = system.set_params(rho=10.0).prepare().system_.elements[0]
    assert element.values['rho'] == pytest.approx(10.0)


def test_dynamical_system_runs_on_the_backend_it_is_given():
    """Test that a dynamical system passes its backend to the system it prepares."""
    trajectory = LorenzSystem().set_params(backend=NumbaBackend()).prepare().simulate(until=1.0, dt=0.5)
    assert trajectory.problem.backend == 'numba'


def test_dynamical_system_equals_the_same_equations_as_an_interaction():
    """Test that the trajectory equals the one of the same equations written as one interaction."""
    rebased = LorenzSystem().prepare().simulate(until=5.0, dt=0.01)
    lorenz = Element('lorenz', values={'x': 1.0, 'y': 1.0, 'z': 1.0, 'sigma': 10.0, 'rho': 28.0, 'beta': 8.0 / 3.0})
    original = System(elements=[lorenz], interactions=[LorenzInteraction('flow', target='lorenz')])
    original = original.prepare().simulate(until=5.0, dt=0.01)
    assert rebased.names == original.names
    for name in original.names:
        np.testing.assert_array_equal(rebased.get_values(name), original.get_values(name))
