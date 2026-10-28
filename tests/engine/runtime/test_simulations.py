"""Test the `_simulations` module of `skcomplex.engine.runtime`."""

import time
from typing import Annotated

import numpy as np
import pytest

from skcomplex.analysis import LoopAnalysis
from skcomplex.core.elements import Element, GroupElement, ParticleElement
from skcomplex.core.errors import ParameterError, SimulationError, SolverError, ValidationError
from skcomplex.core.laws import RateLaw, RuleLaw
from skcomplex.core.spaces import ContinuousSpace
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import BaseObserver, ConservationObserver, RealTimeClock, Simulation
from skcomplex.engine.solvers import GillespieSolver, RK4Solver
from skcomplex.models import create, describe
from skcomplex.models.biology.ecology import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    PopulationElement,
)
from skcomplex.models.biology.epidemiology import CompartmentElement, InfectionInteraction, RecoveryInteraction
from skcomplex.models.economics.finance import geometric_brownian_motion
from skcomplex.models.economics.markets import CobwebInteraction, MarketElement
from skcomplex.models.physics.realizations.particles import ConstantForceInteraction
from skcomplex.models.physics.statistical_mechanics import ideal_gas

from ...conftest import _correlated, _summed


def _pushed_gas(n=500):
    """Return a group pushed by a constant force, whose velocity change depends on the mass."""
    return System(
        elements=[GroupElement('gas', element=ParticleElement, size=n)],
        interactions=[ConstantForceInteraction('push', target='gas', F=[1.0, 0.0])],
        space=ContinuousSpace(n_dim=2),
    )


class FrameCounterObserver(BaseObserver):
    """Observer that keeps the times of the frames it receives."""

    def _prepare(self, problem):
        """Start with no frames."""
        self.times_ = []
        self.stopped_ = False

    def _run(self, frame=None):
        """Keep the time of the frame."""
        if frame is not None:
            self.times_.append(frame.time)

    def _release(self):
        """Mark the end of the run."""
        self.stopped_ = True


def _growth(r=1.0):
    """Return a system of exponential growth."""
    return System(
        elements=[PopulationElement('a', n=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=r)],
    )


def _epidemic():
    """Return a stochastic SIR system."""
    return System(
        elements=[CompartmentElement('S', n=95), CompartmentElement('I', n=5), CompartmentElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.01, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.3, stochastic=True),
        ],
    )


def _logistic():
    """Return a logistic population."""
    return System(
        elements=[PopulationElement('rabbits', n=10)],
        interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=100)],
    )


class SaverElement(Element):
    """Agent that adds its group's rate to its savings every step."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        savings: Annotated[float, StateSpec()] = 0.0,
        rate: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.savings = savings
        self.rate = rate

    def _get_laws(self, s):
        """Return the saving rule."""
        return [RuleLaw(self.save)]

    def save(self, agent, model):
        """Add the rate to the savings."""
        del model
        agent.savings += agent.rate


class HeaterElement(Element):
    """Heater element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[object, StateSpec()] = 0.0,
        power: Annotated[object, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.power = power

    def _get_laws(self, s):
        """Return the laws of the heater element."""
        return [RateLaw(s.T, s.power - s.T)]


STORED_GEOMETRIC_BROWNIAN_MOTION = [
    100.0,
    104.33582683330506,
    105.59332585811273,
    106.6641122066072,
    102.59377817820878,
    99.79646664973221,
    94.25377366277695,
    92.50690616708295,
    92.25856315395363,
    98.3123414995193,
    93.20945394424345,
]


def test_simulation_updates_one_per_entry_weight_live():
    """Test that a live update of one component of a per-entry parameter changes that entry only."""
    schedule = {'sum.weights[1]': lambda t: 1.0 if t < 1 else 10.0}
    trajectory = Simulation(_summed(['a', 'b', 'c']), dt=1, schedule=schedule).run(until=2).trajectory_
    assert trajectory.values('target.x').tolist() == [0.0, 7.0, 25.0]
    simulation = Simulation(_summed(['a', 'b', 'c'])).run(until=1)
    simulation.update_parameters({'sum.weights[2]': 5.0})
    assert simulation.parameters_.tolist() == [1.0, 1.0, 5.0]


def test_closed_economy_changes_money_only_by_lending_and_repayment():
    """Test that total money changes by lending minus repayment, and every transaction-flow row sums to zero."""
    trajectory = (
        Simulation(create('stock_flow_consistent_economy'), dt=1, observers=[ConservationObserver()])
        .run(100)
        .trajectory_
    )
    money = (
        trajectory.values('households.deposits')
        + trajectory.values('firms.deposits')
        + trajectory.values('bank.deposits')
    )
    created = trajectory.values('circuit.lending') - trajectory.values('circuit.repayments')
    assert np.diff(money) == pytest.approx(created[:-1], abs=1e-9)
    for frame in (0, 50, 100):
        matrix = LoopAnalysis('flow-matrix', frame=frame).check().prepare(trajectory).run().matrix_
        assert matrix.sum(axis=1) == pytest.approx(0.0, abs=1e-12)


def test_trajectory_returns_group_arrays():
    """Test that a group state over 100 frames has the shape `(100, 500, 2)`."""
    trajectory = Simulation(_pushed_gas(), dt=0.01).run(until=0.99).trajectory_
    assert trajectory.values('gas.x').shape == (100, 500, 2)
    assert trajectory.values('gas.m').shape == (100, 500)


def test_group_parameter_changes_while_running():
    """Test that setting the mass of every member applies to the frames after the change."""
    simulation = Simulation(_pushed_gas(), dt=0.01, clock=RealTimeClock(speed=1.0)).start(until=0.4)
    time.sleep(0.15)
    simulation.update_parameters({'gas.m': 2.0})
    simulation.wait()
    trajectory = simulation.trajectory_
    changed = int(np.argmax(np.all(trajectory.values('gas.m') == 2.0, axis=1)))
    assert changed > 0
    accelerations = np.diff(trajectory.values('gas.v')[:, :, 0], axis=0) / 0.01
    assert np.allclose(accelerations[changed:], 0.5)
    assert np.allclose(accelerations[: changed - 1], 1.0)


def test_group_parameter_takes_per_member_values():
    """Test that a group parameter accepts one value per member."""
    simulation = Simulation(_pushed_gas(3), dt=0.1).run(until=0.1)
    simulation.update_parameters({'gas.m': [1.0, 2.0, 4.0]})
    assert simulation.problem_.parameters.read(simulation.parameters_, 'gas.m').tolist() == [1.0, 2.0, 4.0]


def test_group_parameter_with_wrong_shape_raises_error():
    """Test that a group parameter rejects values that do not broadcast to the group."""
    simulation = Simulation(_pushed_gas(3), dt=0.1).run(until=0.1)
    with pytest.raises(ValidationError, match='broadcasts'):
        simulation.update_parameters({'gas.m': [1.0, 2.0]})


def test_reflecting_box_keeps_members_inside_ode_run():
    """Test that no member crosses a wall of a reflecting box in an ODE system."""
    trajectory = Simulation(ideal_gas(500, kT=4.0, random_state=1), dt=0.01).run(until=2).trajectory_
    positions = trajectory.values('gas.x')
    assert np.all((positions >= 0) & (positions <= 1))


@pytest.mark.benchmark
def test_ideal_gas_runs_in_real_time():
    """Test that an ideal gas of 10000 molecules keeps a real-time clock within 0.1 seconds."""
    simulation = Simulation(ideal_gas(10_000, random_state=0), dt=0.01, clock=RealTimeClock(speed=1.0), record=False)
    simulation.run(until=5)
    assert simulation.clock_.max_lag_ <= 0.1


def test_simulation_hands_frames_to_observers():
    """Test that observers receive one frame per dt and the end of the run."""
    counter = FrameCounterObserver()
    Simulation(_growth(), dt=0.1, observers=[counter]).run(until=1)
    assert np.allclose(counter.times_, np.linspace(0, 1, 11))
    assert counter.stopped_


def test_simulation_discrete_frames_are_integer_times():
    """Test that discrete maps produce frames at integer times."""
    system = System(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')])
    trajectory = Simulation(system).run(until=10).trajectory_
    assert trajectory.times.tolist() == list(map(float, range(11)))


def test_simulation_discrete_noninteger_dt_raises_error():
    """Test that a discrete map needs an integer dt."""
    system = System(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')])
    with pytest.raises(ValidationError, match='positive integer'):
        Simulation(system, dt=0.5).run(until=1)


def test_simulation_wrong_solver_raises_error():
    """Test that a solver of another form is rejected."""
    with pytest.raises(SolverError, match='GillespieSolver'):
        Simulation(_growth(), solver=GillespieSolver()).run(until=1)


def test_simulation_real_time_paces_frames():
    """Test that a real-time run takes at least its simulated duration."""
    clock = RealTimeClock(speed=1.0)
    start = time.perf_counter()
    Simulation(_growth(), dt=0.1, clock=clock).run(until=0.5)
    assert time.perf_counter() - start >= 0.5
    assert clock.max_lag_ >= 0.0


def test_simulation_update_parameters_while_running():
    """Test that a live parameter change stops the growth of a background run."""
    simulation = Simulation(_growth(), dt=0.01, clock=RealTimeClock(speed=1.0)).start(until=0.6)
    time.sleep(0.2)
    simulation.update_parameters({'g.r': 0.0})
    simulation.wait()
    trajectory = simulation.trajectory_
    rates = trajectory.values('g.r')
    changed = int(np.argmax(rates == 0.0))
    assert changed > 0
    assert np.allclose(trajectory.values('a.n')[changed:], trajectory.values('a.n')[changed])


def test_simulation_update_unknown_parameter_raises_error():
    """Test that only parameters of the problem can be set."""
    simulation = Simulation(_growth()).run(until=0.1)
    with pytest.raises(ParameterError, match=r"'g\.k'"):
        simulation.update_parameters({'g.k': 1.0})


def test_simulation_update_invalid_value_raises_error():
    """Test that parameter values are checked against their domain."""
    system = System(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')])
    simulation = Simulation(system).run(until=1)
    with pytest.raises(ValidationError, match=r'c\.b = -1\.0 is not a value .* in the domain \(0, \+oo\)'):
        simulation.update_parameters({'c.b': -1.0})


def test_simulation_update_after_run_applies_immediately():
    """Test that a change outside a run applies to the current parameters."""
    simulation = Simulation(_growth()).run(until=0.1)
    simulation.update_parameters({'g.r': 2.5})
    assert simulation.parameters_.tolist() == [2.5]


def test_simulation_stop_background_run():
    """Test that an unbounded background run stops and keeps its frames."""
    simulation = Simulation(_growth(r=0.0), dt=0.01, clock=RealTimeClock(speed=1.0)).start()
    time.sleep(0.1)
    simulation.pause()
    simulation.resume()
    simulation.stop()
    assert not simulation._thread.is_alive()
    assert len(simulation.trajectory_.times) > 1


def test_simulation_background_error_raised_on_wait():
    """Test that an error in the background thread is raised by wait."""

    class FailingObserver(BaseObserver):
        """Observer that fails on the second frame."""

        def _run(self, frame=None):
            """Fail after the first frame."""
            if frame is not None and frame.step == 1:
                error_msg = 'failure'
                raise RuntimeError(error_msg)

    simulation = Simulation(_growth(), observers=[FailingObserver()]).start(until=1)
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
        trajectory.values('b.n')


def test_simulation_same_random_state_is_reproducible():
    """Test that a fixed random state reproduces a stochastic run."""
    first = Simulation(_epidemic(), dt=0.5, random_state=0).run(until=20).trajectory_
    second = Simulation(_epidemic(), dt=0.5, random_state=0).run(until=20).trajectory_
    assert np.array_equal(first.states, second.states)


def test_simulation_parameters_carry_over_runs():
    """Test that a growth rate set between runs is used by the next run and recorded in every frame."""
    simulation = Simulation(_logistic(), dt=1.0).run(until=5)
    simulation.update_parameters({'growth.r': 0.0})
    trajectory = simulation.run(until=10).trajectory_
    assert np.all(trajectory.values('rabbits.n') == 10.0)
    assert np.all(trajectory.values('growth.r') == 0.0)


def test_simulation_agent_parameters_carry_over_runs():
    """Test that an agent group parameter set between runs is read by the rules of the next run."""
    simulation = Simulation(System(elements=[GroupElement('savers', element=SaverElement, size=3)])).run(until=0)
    simulation.update_parameters({'savers.rate': 5.0})
    trajectory = simulation.run(until=2).trajectory_
    assert trajectory.agents('savers', 2).values['savings'].tolist() == [10.0, 10.0, 10.0]


def test_simulation_extend_runs_in_two_parts():
    """Test that an extended run appends frames that follow a parameter change."""
    simulation = Simulation(_logistic(), dt=1.0).run(until=5)
    simulation.update_parameters({'growth.r': 0.0})
    trajectory = simulation.extend(until=10).trajectory_
    reference = Simulation(_logistic(), dt=1.0).run(until=5).trajectory_.values('rabbits.n')
    assert trajectory.times.tolist() == [float(time) for time in range(11)]
    assert trajectory.values('rabbits.n')[:6] == pytest.approx(reference)
    assert np.all(trajectory.values('rabbits.n')[5:] == trajectory.values('rabbits.n')[5])


def test_simulation_extend_backwards_raises_error():
    """Test that extending to an earlier time raises a named error that names the current time."""
    simulation = Simulation(_logistic(), dt=1.0).run(until=5)
    with pytest.raises(SimulationError, match=r'current time 5\.0'):
        simulation.extend(until=3)


def test_simulation_extend_before_run_raises_error():
    """Test that extending a simulation that has not run raises a named error."""
    with pytest.raises(SimulationError, match='has not been run'):
        Simulation(_logistic()).extend(until=3)


@pytest.mark.parametrize(('until', 'frames', 'dt'), [(1, 101, 0.01), (1000, 1001, 1.0), (15, 751, 0.02)])
def test_simulation_default_frames(until, frames, dt):
    """Test that default frames are 0.01 apart for short runs and grow in 1, 2, 5 steps to at most 1000 frames."""
    counter = FrameCounterObserver()
    simulation = Simulation(_growth(r=0.0), observers=[counter]).run(until=until)
    assert len(counter.times_) == frames
    assert simulation.dt_ == dt


def test_trajectory_lists_its_names():
    """Test that a trajectory lists the names of its states, parameters, reporters, and observables."""
    system = create('hilsa_fishery')
    observables = describe('hilsa_fishery').observables_of(system)
    trajectory = Simulation(system, dt=1, observables=observables).run(until=2).trajectory_
    names = trajectory.names
    assert 'hilsa.adult_weight' in names
    assert all(name in names for name in trajectory.problem.states.names)


def test_trajectory_gives_agent_locations_over_frames():
    """Test that the locations of a group of agents come as one array per frame."""
    trajectory = Simulation(create('boids'), random_state=0).run(until=3).trajectory_
    locations = trajectory.agent_locations('flock')
    assert len(locations) == len(trajectory.times)
    np.testing.assert_allclose(locations[-1], trajectory.agents('flock', len(trajectory.times) - 1).locations)


def test_initial_state_replaces_the_values_of_components():
    """Test that a simulation starts from given initial values by state name and rejects unknown names."""
    system = System(
        elements=[PopulationElement('rabbits', n=10)],
        interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.0)],
    )
    trajectory = Simulation(system, dt=1.0, initial_state={'rabbits.n': 20.0}).run(until=1.0).trajectory_
    assert trajectory.values('rabbits.n')[0] == pytest.approx(20.0)
    with pytest.raises(ValidationError, match=r'foxes\.n'):
        Simulation(system, dt=1.0, initial_state={'foxes.n': 1.0}).run(until=1.0)


def test_simulation_schedules_a_bounded_control():
    """Test that a scheduled control takes its value at every frame, which laws read and trajectories record."""
    schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
    trajectory = (
        Simulation(System(elements=[HeaterElement('heater')]), dt=0.5, schedule=schedule).run(until=3.0).trajectory_
    )
    np.testing.assert_array_equal(trajectory.values('heater.power'), [0, 0, 1, 1, 1, 1, 1])
    assert trajectory.values('heater.T')[2] == pytest.approx(0.0)
    assert trajectory.values('heater.T')[-1] == pytest.approx(1 - np.exp(-2.0), rel=1e-3)


def test_correlated_noise_repeats_with_same_seed():
    """Test that two runs of a correlated `WienerProcessSpec` with random state 0 on the Numba backend are equal."""
    system = _correlated([[1.0, 0.5], [0.5, 1.0]])
    runs = [
        Simulation(system, backend=NumbaBackend(), random_state=0, dt=0.1).run(until=2).trajectory_.states
        for _ in range(2)
    ]
    np.testing.assert_array_equal(*runs)


def test_perfect_correlation_keeps_states_equal():
    """Test that processes of correlation 1 keep two states with unit noise equal at every frame."""
    trajectory = Simulation(_correlated([[1.0, 1.0], [1.0, 1.0]]), random_state=0, dt=0.1).run(until=2).trajectory_
    np.testing.assert_allclose(trajectory.values('a.n'), trajectory.values('b.n'), atol=1e-12)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_unnamed_noise_reproduces_stored_trajectory(backend):
    """Test that geometric Brownian motion with random state 0 repeats the stored trajectory on both backends."""
    simulation = Simulation(geometric_brownian_motion(), backend=backend(), random_state=0, dt=0.05)
    trajectory = simulation.run(until=0.5).trajectory_
    np.testing.assert_array_equal(trajectory.values('asset.price'), STORED_GEOMETRIC_BROWNIAN_MOTION)


class _ReleaseCountingSolver(RK4Solver):
    """RK4 solver that counts the final runs that release it."""

    def _prepare(self, problem, rng, until=float('inf')):
        """Fit the solver and start with no release."""
        super()._prepare(problem, rng, until)
        self.releases_ = 0

    def _release(self):
        """Count the release."""
        self.releases_ += 1


def test_simulation_releases_the_solver_at_the_last_frame():
    """Test that a run fits the solver once and releases it with the run of its last frame."""
    simulation = Simulation(_growth(), solver=_ReleaseCountingSolver(), dt=0.1).run(until=1)
    assert simulation.stepper_.releases_ == 1
    assert simulation.trajectory_.times.size == 11


def test_simulation_releases_the_solver_after_a_stop():
    """Test that a stopped background run releases its solver once."""
    simulation = Simulation(_growth(r=0.0), solver=_ReleaseCountingSolver(), dt=0.01, clock=RealTimeClock(speed=1.0))
    simulation.start()
    time.sleep(0.05)
    simulation.stop()
    assert simulation.stepper_.releases_ == 1
