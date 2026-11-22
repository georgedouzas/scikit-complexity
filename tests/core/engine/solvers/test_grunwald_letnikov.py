"""Test the `GrunwaldLetnikovSolver` class."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis._symbolic import _MittagLefflerFunction
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import fractional
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import (
    CoSimulationSolver,
    FractionalABMSolver,
    GrunwaldLetnikovSolver,
    L1Solver,
    RK4Solver,
    StepSpec,
)
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System

SCHEMES = [GrunwaldLetnikovSolver, L1Solver, FractionalABMSolver]


def _relaxation(s, alpha):
    """Return the relaxation `-x` of the value `x` of the target, with a Caputo derivative of an order below 1."""
    x = s.target.x
    return RateLaw(fractional(x, alpha) if alpha < 1 else x.diff(s.t), -x)


def _relaxing(label: str = 'a') -> System:
    """Return a value that relaxes with a Caputo derivative of order 0.6."""
    relax = Interaction('relax', roles={'target': 'a'}, laws=lambda s: [_relaxation(s, 0.6)])
    return System(label, elements=[Element('a', values={'x': 1.0})], interactions=[relax])


def _decaying() -> System:
    """Return a value that decays exponentially."""
    decay = Interaction('decay', roles={'target': 'a'}, laws=lambda s: [_relaxation(s, 1)])
    return System(elements=[Element('a', values={'x': 1.0})], interactions=[decay])


def _mixed() -> System:
    """Return a fractional value of order 0.5 and an ordinary value that both relax."""
    relax = Interaction(
        'relax',
        roles={'target': 'a'},
        laws=lambda s: [_relaxation(s, 0.5), RateLaw(s.target.y.diff(s.t), -s.target.y)],
    )
    return System(elements=[Element('a', values={'x': 1.0, 'y': 1.0})], interactions=[relax])


def _counter(label: str) -> System:
    """Return a value that counts the steps of a discrete map."""
    count = Interaction('count', roles={'target': 'c'}, laws=lambda s: [UpdateLaw(s.target.k, s.target.k + 1)])
    return System(label, elements=[Element('c', values={'k': 0.0})], interactions=[count])


def _advance(stepper, problem, until, frames=1):
    """Return the state of a problem after advancing a stepper to a time in equal frames."""
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    x, t = problem.initial_state, 0.0
    for index in range(frames):
        following = until * (index + 1) / frames
        x = stepper.run(StepSpec(t, x, problem.initial_parameters, following)).state_
        t = following
    return x


def test_compiler_rate_law_on_fractional_state_gives_fractional_form():
    """Test that a rate law on a state of order 0.6 compiles to the fractional form with the order of every state."""
    problem = _mixed().prepare()._problem
    assert problem.form == FormKind.FRACTIONAL
    assert problem.orders.tolist() == [0.5, 1.0]
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [-1.0, -1.0]


def test_compiler_order_one_gives_ode_form():
    """Test that a state of order 1 compiles to the ODE form without orders."""
    problem = _decaying().prepare()._problem
    assert problem.form == FormKind.ODE
    assert problem.orders.size == 0


@pytest.mark.parametrize('scheme', SCHEMES)
@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend], ids=['numpy', 'numba'])
def test_scheme_converges_to_mittag_leffler_relaxation(scheme, backend):
    """Test that a scheme matches E_0.6(-t^0.6) at time 5, with a smaller error for the smaller step."""
    problem = _relaxing().set_params(backend=backend()).prepare()._problem
    exact = float(_MittagLefflerFunction._get_mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    errors = [abs(_advance(scheme(dt=dt), problem, 5.0)[0] - exact) for dt in (0.01, 0.005)]
    assert errors[0] < 1e-3
    assert errors[1] < errors[0]


@pytest.mark.parametrize('scheme', [GrunwaldLetnikovSolver, L1Solver])
def test_scheme_short_memory_matches_full_memory(scheme):
    """Test that a memory window of 10 time units stays within 1e-3 of the whole history at time 20."""
    problem = _relaxing().prepare()._problem
    full = _advance(scheme(dt=0.01), problem, 20.0)
    short = _advance(scheme(dt=0.01, memory=10.0), problem, 20.0, frames=20)
    assert abs(short[0] - full[0]) < 1e-3


@pytest.mark.parametrize('scheme', SCHEMES)
def test_scheme_of_order_one_matches_exponential_decay(scheme):
    """Test that a scheme on a state of order 1 follows the exponential decay."""
    problem = _decaying().prepare()._problem
    assert _advance(scheme(dt=0.001), problem, 1.0)[0] == pytest.approx(math.exp(-1.0), abs=1e-3)


@pytest.mark.parametrize('scheme', SCHEMES)
def test_scheme_frames_continue_the_history(scheme):
    """Test that advancing in several frames gives the run of one frame, with ordinary states along."""
    problem = _mixed().prepare()._problem
    whole = _advance(scheme(dt=0.01), problem, 2.0)
    framed = _advance(scheme(dt=0.01), problem, 2.0, frames=8)
    assert framed == pytest.approx(whole, abs=1e-12)
    assert framed[1] == pytest.approx(math.exp(-2.0), abs=1e-2)


def test_scheme_restarts_after_a_change_of_a_fractional_state():
    """Test that a change of a fractional state between frames starts a new history from the changed state."""
    problem = _relaxing().prepare()._problem
    stepper = GrunwaldLetnikovSolver(dt=0.01)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    p = problem.initial_parameters
    stepper.run(StepSpec(0.0, problem.initial_state, p, 1.0))
    restarted = stepper.run(StepSpec(1.0, np.array([1.0]), p, 2.0)).state_
    fresh = _advance(GrunwaldLetnikovSolver(dt=0.01), problem, 1.0)
    assert restarted == pytest.approx(fresh, abs=1e-12)


def test_scheme_rejects_times_off_the_grid():
    """Test that a frame that does not fall on the grid of the run is rejected."""
    problem = _relaxing().prepare()._problem
    stepper = L1Solver(dt=0.1)
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    with pytest.raises(SolverError, match='multiple of dt'):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 0.25))


@pytest.mark.parametrize(('stepper', 'match'), [(L1Solver(dt=0.0), 'dt'), (L1Solver(memory=-1.0), 'memory')])
def test_scheme_rejects_invalid_settings(stepper, match):
    """Test that a step or a memory window that is not positive is rejected."""
    problem = _relaxing().prepare()._problem
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    with pytest.raises(SolverError, match=match):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))


def test_ordinary_stepper_rejects_fractional_form():
    """Test that an ODE stepper rejects a problem of the fractional form."""
    problem = _relaxing().prepare()._problem
    with pytest.raises(SolverError, match='fractional differential equations'):
        RK4Solver(random_state=0).prepare(problem)


def test_simulation_runs_fractional_system_with_default_solver():
    """Test that a simulation runs a fractional system with the fractional predictor-corrector by default."""
    trajectory = _relaxing().prepare().simulate(until=5.0, dt=0.5)
    assert DEFAULT_SOLVERS[trajectory.problem.form] is FractionalABMSolver
    exact = float(_MittagLefflerFunction._get_mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    assert trajectory.get_values('a.x')[-1] == pytest.approx(exact, abs=1e-4)


def test_cosimulation_runs_fractional_partition():
    """Test that a fractional subsystem runs as a partition of the fractional form next to a discrete map."""
    system = System(systems=[_relaxing('memory'), _counter('steps')], solver=CoSimulationSolver())
    trajectory = system.prepare().simulate(until=5.0, dt=1.0)
    forms = {partition.name: partition.problem.form for partition in trajectory.problem.partitions}
    assert forms == {'memory': FormKind.FRACTIONAL, 'steps': FormKind.DISCRETE}
    exact = float(_MittagLefflerFunction._get_mittag_leffler(0.6, 1.0, -(5.0**0.6)))
    assert trajectory.get_values('memory/a.x')[-1] == pytest.approx(exact, abs=1e-4)
    assert trajectory.get_values('steps/c.k')[-1] == 5
