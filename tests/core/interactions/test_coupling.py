"""Test the `_coupling` module of `skcomplex.core.interactions`."""

import math
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import scipy.optimize
from sage.all__sagemath_symbolics import sin

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.interactions import CouplingInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.spaces import LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
GAMMA = 0.5


class OscillatorElement(Element):
    """Oscillator with a phase and a natural frequency."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        """Return the rotation at the natural frequency."""
        return [RateLaw(s.theta, s.omega)]


def _sine(other, member):
    """Return the sine of the phase difference."""
    return sin(other - member)


def _system(space, theta, omega, function=_sine, **arguments):
    """Return oscillators on a space coupled by a function, the sine of the phase difference by default."""
    oscillators = LatticeElement(
        'oscillators',
        element=OscillatorElement,
        space=space,
        values={'theta': theta, 'omega': omega},
    )
    coupling = CouplingInteraction('coupling', nodes='oscillators', state='theta', function=function, **arguments)
    return System(elements=[oscillators], interactions=[coupling])


def _phases(system, until, backend=None):
    """Return the phases of the oscillators at every unit of time."""
    simulation = Simulation(system, dt=1.0, solver=RK4Solver(dt=0.05), backend=backend).check().prepare()
    return simulation.run(until).trajectory_.values('oscillators.theta')


def _drift(system, backend):
    """Return the rates of the phases at the initial state."""
    problem = compile_system(system, backend=backend)
    return problem.states.read(
        problem.drift(0.0, problem.initial_state, problem.initial_parameters),
        'oscillators.theta',
    )


def _order(phases):
    """Return the modulus of the mean phase vector of every frame."""
    return np.abs(np.exp(1j * phases).mean(axis=1))


@BACKENDS
@pytest.mark.parametrize('mode', ['edges', 'mean-field'])
def test_kuramoto_synchronizes_above_the_critical_strength(mode, backend):
    """Test that 500 oscillators on a complete graph coupled above the critical strength reach an order above 0.8."""
    size = 500
    omega = GAMMA * np.tan(math.pi * ((np.arange(size) + 0.5) / size - 0.5))
    theta = np.random.default_rng(1).uniform(0, 2 * math.pi, size)
    strength = 8 * GAMMA
    system = _system(
        NetworkSpace(nx.complete_graph(size)),
        theta,
        omega,
        K=strength / size if mode == 'edges' else strength,
        mode=mode,
    )
    order = _order(_phases(system, 20.0, backend))
    assert order[0] < 0.1
    assert order[-5:].min() > 0.8


@BACKENDS
def test_coupling_by_difference_diffuses_along_the_bonds(backend):
    """Test that the default coupling adds `K (x_j - x_i)` for every neighbour along both directions of a bond."""
    theta = np.array([1.0, 0.0, 0.0, 3.0])
    system = _system(LatticeSpace(4), theta, np.zeros(4), function=None, K=2.0)
    assert _drift(system, backend).tolist() == [2.0, 2.0, 6.0, -10.0]


@BACKENDS
def test_mean_field_is_the_mean_over_every_member(backend):
    """Test that the mean-field coupling is the mean of the coupling to every member, the member included."""
    theta = np.random.default_rng(2).uniform(0, 2 * math.pi, 6)
    system = _system(NetworkSpace(nx.path_graph(6)), theta, np.zeros(6), K=1.5, alpha=0.3, mode='mean-field')
    expected = 1.5 * np.sin(theta[None, :] - theta[:, None] - 0.3).mean(axis=1)
    assert np.allclose(_drift(system, backend), expected, atol=1e-12)


def test_phase_lag_shifts_the_frequency_of_synchrony():
    """Test that identical oscillators with a phase lag lock at the frequency `omega - K sin(alpha)`."""
    theta = np.random.default_rng(3).uniform(-0.3, 0.3, 10)
    system = _system(NetworkSpace(nx.complete_graph(10)), theta, np.ones(10), K=1.0, alpha=0.5, mode='mean-field')
    phases = _phases(system, 40.0)
    assert np.ptp(phases[-1]) < 1e-6
    assert phases[-1, 0] - phases[-2, 0] == pytest.approx(1 - math.sin(0.5), abs=1e-6)


def test_delayed_coupling_locks_two_oscillators_at_the_delayed_frequency():
    """Test that two delayed oscillators lock in phase at the frequency `Omega` that solves `Omega = 1 - sin(Omega)`."""
    system = _system(NetworkSpace(nx.path_graph(2)), np.array([0.0, 0.5]), np.ones(2), K=1.0, tau=1.0)
    phases = _phases(system, 60.0)
    frequency = scipy.optimize.brentq(lambda value: value - 1 + math.sin(value), 0.0, 1.0)
    assert abs(phases[-1, 1] - phases[-1, 0]) < 1e-4
    assert phases[-1, 0] - phases[-2, 0] == pytest.approx(frequency, abs=1e-4)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'mode': 'chain'}, 'mode of coupling'), ({'function': 1.0}, 'callable')],
)
def test_check_rejects_unknown_modes_and_functions(arguments, match):
    """Test that an unknown mode or a function that cannot be called raises an error."""
    with pytest.raises(ValidationError, match=match):
        CouplingInteraction('coupling', nodes='oscillators', **arguments).check()
