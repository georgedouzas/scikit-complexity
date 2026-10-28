"""Test the steppers of Caputo fractional differential equations and the fractional form they solve."""

import math
from typing import Annotated

import numpy as np
import pytest

from skcomplex.analysis._fractional import _mittag_leffler
from skcomplex.core.elements import Element
from skcomplex.core.errors import SolverError
from skcomplex.core.laws import RateLaw, UpdateLaw
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.problems import Form
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import (
    CoSimulationSolver,
    FractionalABMSolver,
    GrunwaldLetnikovSolver,
    L1Solver,
    RK4Solver,
)

SCHEMES = [GrunwaldLetnikovSolver, L1Solver, FractionalABMSolver]


class RelaxingElement(Element):
    """Element whose state relaxes with a Caputo derivative of order 0.6."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=0.6)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the relaxation rate."""
        return [RateLaw(s.x, -s.x)]


class DecayingElement(Element):
    """Element whose state decays exponentially."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=1)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the decay rate."""
        return [RateLaw(s.x, -s.x)]


class MixedElement(Element):
    """Element with a fractional state and an ordinary state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=0.5)] = 1.0,
        y: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y

    def _get_laws(self, s):
        """Return the relaxation of both states."""
        return [RateLaw(s.x, -s.x), RateLaw(s.y, -s.y)]


class CounterElement(Element):
    """Element whose state counts the steps of a discrete map."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k

    def _get_laws(self, s):
        """Return the count."""
        return [UpdateLaw(s.k, s.k + 1)]


def _advance(stepper, problem, until, frames=1):
    """Return the state of a problem after advancing a stepper to a time in equal frames."""
    stepper.prepare(problem, np.random.default_rng(0))
    x, t = problem.initial_state, 0.0
    for index in range(frames):
        following = until * (index + 1) / frames
        x = stepper.run(t, x, problem.initial_parameters, following)
        t = following
    return x


def test_compile_system_rate_law_on_fractional_state_gives_fractional_form():
    """Test that a rate law on a state of order 0.6 compiles to the fractional form with the order of every state."""
    problem = compile_system(System(elements=[MixedElement('a')]))
    assert problem.form == Form.FRACTIONAL
    assert problem.orders.tolist() == [0.5, 1.0]
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [-1.0, -1.0]


def test_compile_system_order_one_gives_ode_form():
    """Test that a state of order 1 compiles to the ODE form without orders."""
    problem = compile_system(System(elements=[DecayingElement('a')]))
    assert problem.form == Form.ODE
    assert problem.orders.size == 0


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend], ids=['numpy', 'numba'])
def test_scheme_converges_to_mittag_leffler_relaxation(scheme, backend):
    """Test that a scheme matches E_0.6(-t^0.6) at time 5, with a smaller error for the smaller step."""
    problem = compile_system(System(elements=[RelaxingElement('a')]), backend=backend())
    exact = float(_mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    errors = [abs(_advance(scheme(dt=dt), problem, 5.0)[0] - exact) for dt in (0.01, 0.005)]
    assert errors[0] < 1e-3
    assert errors[1] < errors[0]


@pytest.mark.parametrize('scheme', [GrunwaldLetnikovSolver, L1Solver])
def test_scheme_short_memory_matches_full_memory(scheme):
    """Test that a memory window of 10 time units stays within 1e-3 of the whole history at time 20."""
    problem = compile_system(System(elements=[RelaxingElement('a')]))
    full = _advance(scheme(dt=0.01), problem, 20.0)
    short = _advance(scheme(dt=0.01, memory=10.0), problem, 20.0, frames=20)
    assert abs(short[0] - full[0]) < 1e-3


@pytest.mark.parametrize('scheme', SCHEMES)
def test_scheme_of_order_one_matches_exponential_decay(scheme):
    """Test that a scheme on a state of order 1 follows the exponential decay."""
    problem = compile_system(System(elements=[DecayingElement('a')]))
    assert _advance(scheme(dt=0.001), problem, 1.0)[0] == pytest.approx(math.exp(-1.0), abs=1e-3)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_scheme_frames_continue_the_history(scheme):
    """Test that advancing in several frames gives the run of one frame, with ordinary states along."""
    problem = compile_system(System(elements=[MixedElement('a')]))
    whole = _advance(scheme(dt=0.01), problem, 2.0)
    framed = _advance(scheme(dt=0.01), problem, 2.0, frames=8)
    assert framed == pytest.approx(whole, abs=1e-12)
    assert framed[1] == pytest.approx(math.exp(-2.0), abs=1e-2)


def test_scheme_restarts_after_a_change_of_a_fractional_state():
    """Test that a change of a fractional state between frames starts a new history from the changed state."""
    problem = compile_system(System(elements=[RelaxingElement('a')]))
    stepper = GrunwaldLetnikovSolver(dt=0.01)
    stepper.prepare(problem, np.random.default_rng(0))
    p = problem.initial_parameters
    stepper.run(0.0, problem.initial_state, p, 1.0)
    restarted = stepper.run(1.0, np.array([1.0]), p, 2.0)
    fresh = _advance(GrunwaldLetnikovSolver(dt=0.01), problem, 1.0)
    assert restarted == pytest.approx(fresh, abs=1e-12)


def test_scheme_rejects_times_off_the_grid():
    """Test that a frame that does not fall on the grid of the run is rejected."""
    problem = compile_system(System(elements=[RelaxingElement('a')]))
    stepper = L1Solver(dt=0.1)
    stepper.prepare(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match='multiple of dt'):
        stepper.run(0.0, problem.initial_state, problem.initial_parameters, 0.25)


@pytest.mark.parametrize(('stepper', 'match'), [(L1Solver(dt=0.0), 'dt'), (L1Solver(memory=-1.0), 'memory')])
def test_scheme_rejects_invalid_settings(stepper, match):
    """Test that a step or a memory window that is not positive is rejected."""
    problem = compile_system(System(elements=[RelaxingElement('a')]))
    stepper.prepare(problem, np.random.default_rng(0))
    with pytest.raises(SolverError, match=match):
        stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0)


def test_ordinary_stepper_rejects_fractional_form():
    """Test that an ODE stepper rejects a problem of the fractional form."""
    problem = compile_system(System(elements=[RelaxingElement('a')]))
    with pytest.raises(SolverError, match='fractional differential equations'):
        RK4Solver().prepare(problem, np.random.default_rng(0))


def test_simulation_runs_fractional_system_with_default_solver():
    """Test that a simulation runs a fractional system with the fractional predictor-corrector by default."""
    simulation = Simulation(System(elements=[RelaxingElement('a')]), dt=0.5).run(until=5.0)
    assert isinstance(simulation.stepper_, FractionalABMSolver)
    exact = float(_mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    assert simulation.trajectory_.values('a.x')[-1] == pytest.approx(exact, abs=1e-4)


def test_cosimulation_runs_fractional_partition():
    """Test that a fractional subsystem runs as a partition of the fractional form next to a discrete map."""
    system = System(
        systems=[System('memory', elements=[RelaxingElement('a')]), System('steps', elements=[CounterElement('c')])],
    )
    simulation = Simulation(system, solver=CoSimulationSolver(), dt=1.0).run(until=5.0)
    forms = {partition.name: partition.problem.form for partition in simulation.problem_.partitions}
    assert forms == {'memory': Form.FRACTIONAL, 'steps': Form.DISCRETE}
    exact = float(_mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    assert simulation.trajectory_.values('memory/a.x')[-1] == pytest.approx(exact, abs=1e-4)
    assert simulation.trajectory_.values('steps/c.k')[-1] == 5
