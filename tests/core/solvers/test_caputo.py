"""Test the steppers of Caputo fractional differential equations and the fractional form they solve."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import mittag_leffler
from skcomplex.core.base import Form, SolverError, state
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Element
from skcomplex.core.laws import Rate, Update
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, CoSimulation, FractionalABM, GrunwaldLetnikov, L1Scheme
from skcomplex.core.system import System

SCHEMES = [GrunwaldLetnikov, L1Scheme, FractionalABM]


class Relaxing(Element):
    """Element whose state relaxes with a Caputo derivative of order 0.6."""

    x: float = state(1.0, order=0.6)

    def laws(self, s):
        """Return the relaxation rate."""
        return [Rate(s.x, -s.x)]


class Decaying(Element):
    """Element whose state decays exponentially."""

    x: float = state(1.0, order=1)

    def laws(self, s):
        """Return the decay rate."""
        return [Rate(s.x, -s.x)]


class Mixed(Element):
    """Element with a fractional state and an ordinary state."""

    x: float = state(1.0, order=0.5)
    y: float = state(1.0)

    def laws(self, s):
        """Return the relaxation of both states."""
        return [Rate(s.x, -s.x), Rate(s.y, -s.y)]


class Counter(Element):
    """Element whose state counts the steps of a discrete map."""

    k: float = state(0.0)

    def laws(self, s):
        """Return the count."""
        return [Update(s.k, s.k + 1)]


def _advance(stepper, problem, until, frames=1):
    """Return the state of a problem after advancing a stepper to a time in equal frames."""
    stepper.initialize(problem, np.random.default_rng(0))
    x, t = problem.initial_state, 0.0
    for index in range(frames):
        following = until * (index + 1) / frames
        x = stepper.advance(t, x, problem.initial_parameters, following)
        t = following
    return x


def test_compile_system_rate_law_on_fractional_state_gives_fractional_form():
    """Test that a rate law on a state of order 0.6 compiles to the fractional form with the order of every state."""
    problem = compile_system(System(components=[Mixed('a')]))
    assert problem.form is Form.FRACTIONAL
    assert problem.orders.tolist() == [0.5, 1.0]
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [-1.0, -1.0]


def test_compile_system_order_one_gives_ode_form():
    """Test that a state of order 1 compiles to the ODE form without orders."""
    problem = compile_system(System(components=[Decaying('a')]))
    assert problem.form is Form.ODE
    assert problem.orders.size == 0


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_scheme_converges_to_mittag_leffler_relaxation(scheme, backend):
    """Test that a scheme matches E_0.6(-t^0.6) at time 5, with a smaller error for the smaller step."""
    problem = compile_system(System(components=[Relaxing('a')]), backend=backend)
    exact = float(mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    errors = [abs(_advance(scheme(dt=dt), problem, 5.0)[0] - exact) for dt in (0.01, 0.005)]
    assert errors[0] < 1e-3
    assert errors[1] < errors[0]


@pytest.mark.parametrize('scheme', [GrunwaldLetnikov, L1Scheme])
def test_scheme_short_memory_matches_full_memory(scheme):
    """Test that a memory window of 10 time units stays within 1e-3 of the whole history at time 20."""
    problem = compile_system(System(components=[Relaxing('a')]))
    full = _advance(scheme(dt=0.01), problem, 20.0)
    short = _advance(scheme(dt=0.01, memory=10.0), problem, 20.0, frames=20)
    assert abs(short[0] - full[0]) < 1e-3


@pytest.mark.parametrize('scheme', SCHEMES)
def test_scheme_of_order_one_matches_exponential_decay(scheme):
    """Test that a scheme on a state of order 1 follows the exponential decay."""
    problem = compile_system(System(components=[Decaying('a')]))
    assert _advance(scheme(dt=0.001), problem, 1.0)[0] == pytest.approx(math.exp(-1.0), abs=1e-3)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_scheme_frames_continue_the_history(scheme):
    """Test that advancing in several frames gives the run of one frame, with ordinary states along."""
    problem = compile_system(System(components=[Mixed('a')]))
    whole = _advance(scheme(dt=0.01), problem, 2.0)
    framed = _advance(scheme(dt=0.01), problem, 2.0, frames=8)
    assert framed == pytest.approx(whole, abs=1e-12)
    assert framed[1] == pytest.approx(math.exp(-2.0), abs=1e-2)


def test_scheme_restarts_after_a_change_of_a_fractional_state():
    """Test that a change of a fractional state between frames starts a new history from the changed state."""
    problem = compile_system(System(components=[Relaxing('a')]))
    stepper = GrunwaldLetnikov(dt=0.01)
    stepper.initialize(problem, np.random.default_rng(0))
    p = problem.initial_parameters
    stepper.advance(0.0, problem.initial_state, p, 1.0)
    restarted = stepper.advance(1.0, np.array([1.0]), p, 2.0)
    fresh = _advance(GrunwaldLetnikov(dt=0.01), problem, 1.0)
    assert restarted == pytest.approx(fresh, abs=1e-12)


def test_scheme_rejects_times_off_the_grid():
    """Test that a frame that does not fall on the grid of the run is rejected."""
    problem = compile_system(System(components=[Relaxing('a')]))
    stepper = L1Scheme(dt=0.1)
    stepper.initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match='multiple of dt'):
        stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 0.25)


@pytest.mark.parametrize(('stepper', 'match'), [(L1Scheme(dt=0.0), 'dt'), (L1Scheme(memory=-1.0), 'memory')])
def test_scheme_rejects_invalid_settings(stepper, match):
    """Test that a step or a memory window that is not positive is rejected."""
    problem = compile_system(System(components=[Relaxing('a')]))
    stepper.initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match=match):
        stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)


def test_ordinary_stepper_rejects_fractional_form():
    """Test that an ODE stepper rejects a problem of the fractional form."""
    problem = compile_system(System(components=[Relaxing('a')]))
    with pytest.raises(SolverError, match='fractional differential equations'):
        RK4().initialize(problem, np.random.default_rng(0))


def test_simulation_runs_fractional_system_with_default_solver():
    """Test that a simulation runs a fractional system with the fractional predictor-corrector by default."""
    simulation = Simulation(System(components=[Relaxing('a')]), dt=0.5).run(until=5.0)
    assert isinstance(simulation.stepper_, FractionalABM)
    exact = float(mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    assert simulation.trajectory_['a.x'][-1] == pytest.approx(exact, abs=1e-4)


def test_cosimulation_runs_fractional_partition():
    """Test that a fractional subsystem runs as a partition of the fractional form next to a discrete map."""
    system = System(
        components=[
            System('memory', components=[Relaxing('a')]),
            System('steps', components=[Counter('c')]),
        ],
    )
    simulation = Simulation(system, solver=CoSimulation(), dt=1.0).run(until=5.0)
    forms = {partition.name: partition.problem.form for partition in simulation.problem_.partitions}
    assert forms == {'memory': Form.FRACTIONAL, 'steps': Form.DISCRETE}
    exact = float(mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    assert simulation.trajectory_['memory/a.x'][-1] == pytest.approx(exact, abs=1e-4)
    assert simulation.trajectory_['steps/c.k'][-1] == 5
