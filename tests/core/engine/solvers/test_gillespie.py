"""Test the `GillespieSolver` class."""

import time

import numpy as np
import pytest

from skcomplex.core.components.elements import NonnegativeElement
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import GillespieSolver
from skcomplex.core.specs import StepSpec
from skcomplex.systems.biology.epidemiology import InfectionInteraction, RecoveryInteraction, SIRSystem


def test_gillespie_conserves_individuals():
    """Test that recoveries move individuals without creating or destroying them."""
    system = System(
        elements=[NonnegativeElement('I', x=50), NonnegativeElement('R')],
        interactions=[RecoveryInteraction('rec', infected='I', recovered='R', gamma=0.5, stochastic=True)],
    )
    problem = PassCompiler().prepare(system).run().problem_
    stepper = GillespieSolver()
    stepper.set_params(random_state=np.random.default_rng(1)).prepare(problem)
    state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_
    assert state.sum() == 50
    assert 0 < state[0] < 50
    assert np.all(state == np.round(state))


def _epidemic():
    """Return a stochastic epidemic of 1000 individuals."""
    return System(
        elements=[NonnegativeElement('S', x=990), NonnegativeElement('I', x=10), NonnegativeElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.0003, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )


def _gillespie_run(backend):
    """Return the duration and the trajectory of a compiled Gillespie run after a warm-up run."""
    simulate(_epidemic(), until=50, backend=backend, solver=GillespieSolver(), dt=1.0, random_state=0)
    start = time.perf_counter()
    trajectory = simulate(_epidemic(), until=50, backend=backend, solver=GillespieSolver(), dt=1.0, random_state=0)
    return time.perf_counter() - start, trajectory


@pytest.mark.benchmark
def test_compiled_gillespie_runs_serial_kernels():
    """Test that the default Numba backend runs a compiled Gillespie epidemic as fast as a serial one."""
    default, first = _gillespie_run(NumbaBackend())
    serial, second = _gillespie_run(NumbaBackend(parallel=False))
    assert np.array_equal(first.states, second.states)
    assert default <= 1.5 * serial + 0.005


def _sir_firings():
    """Return a recorded Gillespie run of the stochastic SIR system and the law of infection."""
    trajectory = simulate(SIRSystem(stochastic=True), until=60, random_state=1, record_draws=True)
    problem = trajectory.problem
    rates = problem.propensities(0.0, problem.initial_state, problem.initial_parameters)
    infection = int(np.argmax(np.isclose(rates, 0.0003 * 990 * 10)))
    return trajectory, infection


def test_gillespie_records_firings_that_rebuild_the_states():
    """Test that the time and law of every firing rebuild the recorded states."""
    trajectory, infection = _sir_firings()
    draws = trajectory.draws
    assert draws.firing_times.size == draws.firing_laws.size > 0
    assert np.all(np.diff(draws.firing_times) >= 0)
    for frame, now in enumerate(trajectory.times):
        fired = draws.firing_laws[draws.firing_times <= now]
        infections, recoveries = np.sum(fired == infection), np.sum(fired != infection)
        assert trajectory.values('S.x')[frame] == 990 - infections
        assert trajectory.values('I.x')[frame] == 10 + infections - recoveries
        assert trajectory.values('R.x')[frame] == recoveries


def test_gillespie_replay_gives_the_path_log_likelihood_at_other_rates():
    """Test the replayed log-likelihood against the closed form of a jump process path."""
    trajectory, infection = _sir_firings()
    beta, gamma = 0.0004, 0.15
    replayed = simulate(SIRSystem(stochastic=True, beta=beta, gamma=gamma), until=60, replay=trajectory)
    np.testing.assert_array_equal(replayed.states, trajectory.states)
    susceptible, infected, last, expected = 990.0, 10.0, 0.0, 0.0
    for fired_at, law in zip(trajectory.draws.firing_times, trajectory.draws.firing_laws, strict=True):
        rates = {infection: beta * susceptible * infected, 1 - infection: gamma * infected}
        expected += np.log(rates[law]) - sum(rates.values()) * (fired_at - last)
        last = fired_at
        susceptible, infected = (susceptible - 1, infected + 1) if law == infection else (susceptible, infected - 1)
    expected -= (beta * susceptible * infected + gamma * infected) * (60 - last)
    assert replayed.log_probability == pytest.approx(expected)
