"""Test simulations, clocks, observers, and trajectories."""

import time

import numpy as np
import pytest

from skcomplex.core.base import ParameterError, SimulationError, SolverError, ValidationError
from skcomplex.core.model import System
from skcomplex.core.runtime import Observer, RealTime, Simulation
from skcomplex.core.solvers import Gillespie
from skcomplex.models.ecology import ExponentialGrowth, Population
from skcomplex.models.economics import Cobweb, Market
from skcomplex.models.epidemiology import Compartment, Infection, Recovery


class FrameCounter(Observer):
    """Observer that keeps the times of the frames it receives."""

    def __init__(self):
        """Start with no frames."""
        self.times = []
        self.stopped = False

    def on_frame(self, frame):
        """Keep the time of the frame."""
        self.times.append(frame.time)

    def on_stop(self):
        """Mark the end of the run."""
        self.stopped = True


def _growth(r=1.0):
    """Return a system of exponential growth."""
    return System(elements=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=r)])


def _epidemic():
    """Return a stochastic SIR system."""
    return System(
        elements=[Compartment('S', n=95), Compartment('I', n=5), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.01, stochastic=True),
            Recovery('recovery', infected='I', recovered='R', gamma=0.3, stochastic=True),
        ],
    )


def test_simulation_hands_frames_to_observers():
    """Test that observers receive one frame per dt and the end of the run."""
    counter = FrameCounter()
    Simulation(_growth(), dt=0.1, observers=[counter]).run(until=1)
    assert np.allclose(counter.times, np.linspace(0, 1, 11))
    assert counter.stopped


def test_simulation_discrete_frames_are_integer_times():
    """Test that discrete maps produce frames at integer times."""
    system = System(elements=[Market('m')], interactions=[Cobweb('c', target='m')])
    trajectory = Simulation(system).run(until=10).trajectory_
    assert trajectory.times.tolist() == list(map(float, range(11)))


def test_simulation_discrete_noninteger_dt_raises_error():
    """Test that a discrete map needs an integer dt."""
    system = System(elements=[Market('m')], interactions=[Cobweb('c', target='m')])
    with pytest.raises(ValidationError, match='positive integer'):
        Simulation(system, dt=0.5).run(until=1)


def test_simulation_wrong_solver_raises_error():
    """Test that a solver of another form is rejected."""
    with pytest.raises(SolverError, match='Gillespie'):
        Simulation(_growth(), solver=Gillespie()).run(until=1)


def test_simulation_real_time_paces_frames():
    """Test that a real-time run takes at least its simulated duration."""
    clock = RealTime(speed=1.0)
    start = time.perf_counter()
    Simulation(_growth(), dt=0.1, clock=clock).run(until=0.5)
    assert time.perf_counter() - start >= 0.5
    assert clock.max_lag_ >= 0.0


def test_simulation_update_parameters_while_running():
    """Test that a live parameter change stops the growth of a background run."""
    simulation = Simulation(_growth(), dt=0.01, clock=RealTime(speed=1.0)).start(until=0.6)
    time.sleep(0.2)
    simulation.update_parameters({'g.r': 0.0})
    simulation.wait()
    trajectory = simulation.trajectory_
    rates = trajectory['g.r']
    changed = int(np.argmax(rates == 0.0))
    assert changed > 0
    assert np.allclose(trajectory['a.n'][changed:], trajectory['a.n'][changed])


def test_simulation_update_unknown_parameter_raises_error():
    """Test that only parameters of the problem can be set."""
    simulation = Simulation(_growth()).run(until=0.1)
    with pytest.raises(ParameterError, match=r"'g\.k'"):
        simulation.update_parameters({'g.k': 1.0})


def test_simulation_update_invalid_value_raises_error():
    """Test that parameter values are checked against their domain."""
    system = System(elements=[Market('m')], interactions=[Cobweb('c', target='m')])
    simulation = Simulation(system).run(until=1)
    with pytest.raises(ValidationError, match='positive'):
        simulation.update_parameters({'c.b': -1.0})


def test_simulation_update_after_run_applies_immediately():
    """Test that a change outside a run applies to the current parameters."""
    simulation = Simulation(_growth()).run(until=0.1)
    simulation.update_parameters({'g.r': 2.5})
    assert simulation.parameters_.tolist() == [2.5]


def test_simulation_stop_background_run():
    """Test that an unbounded background run stops and keeps its frames."""
    simulation = Simulation(_growth(r=0.0), dt=0.01, clock=RealTime(speed=1.0)).start()
    time.sleep(0.1)
    simulation.pause()
    simulation.resume()
    simulation.stop()
    assert not simulation._thread.is_alive()
    assert len(simulation.trajectory_) > 1


def test_simulation_background_error_raised_on_wait():
    """Test that an error in the background thread is raised by wait."""

    class Failing(Observer):
        """Observer that fails on the second frame."""

        def on_frame(self, frame):
            """Fail after the first frame."""
            if frame.step == 1:
                error_msg = 'failure'
                raise RuntimeError(error_msg)

    simulation = Simulation(_growth(), observers=[Failing()]).start(until=1)
    with pytest.raises(RuntimeError, match='failure'):
        simulation.wait()


def test_simulation_methods_before_run_raise_error():
    """Test that control methods need a run first."""
    with pytest.raises(SimulationError, match='has not been run'):
        Simulation(_growth()).update_parameters({'g.r': 1.0})


def test_simulation_without_recording_has_no_trajectory():
    """Test that disabling recording removes the trajectory."""
    simulation = Simulation(_growth(), record=False).run(until=0.1)
    with pytest.raises(SimulationError, match='record=True'):
        _ = simulation.trajectory_


def test_trajectory_unknown_name_raises_error():
    """Test that a trajectory names the available variables."""
    trajectory = Simulation(_growth()).run(until=0.1).trajectory_
    with pytest.raises(KeyError, match=r'a\.n, g\.r'):
        trajectory['b.n']


def test_simulation_same_random_state_is_reproducible():
    """Test that a fixed random state reproduces a stochastic run."""
    first = Simulation(_epidemic(), dt=0.5, random_state=0).run(until=20).trajectory_
    second = Simulation(_epidemic(), dt=0.5, random_state=0).run(until=20).trajectory_
    assert np.array_equal(first.states, second.states)
