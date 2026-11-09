"""Test the `_simulations` module of `skcomplex.core.engine.runtime`."""

import inspect
import time
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from skcomplex.core.analysis import loops
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, RuleLaw, TransitionLaw, UpdateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import BaseObserver, ConservationObserver, RealTimeClock, simulate
from skcomplex.core.engine.runtime._simulations import Simulation
from skcomplex.core.engine.solvers import GillespieSolver, RK4Solver, ScipyODESolver
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import ParameterError, SimulationError, SolverError, ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.collective_behaviour import BoidsSystem
from skcomplex.systems.biology.common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    PopulationElement,
)
from skcomplex.systems.biology.epidemiology import InfectionInteraction, RecoveryInteraction
from skcomplex.systems.computer_science.automata import GameOfLifeSystem
from skcomplex.systems.economics.finance import GeometricBrownianMotionSystem
from skcomplex.systems.economics.macroeconomics import StockFlowConsistentEconomySystem
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.economics.resources import HilsaFisherySystem
from skcomplex.systems.physics.common.particles import ConstantForceInteraction, MassPointElement
from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem

from ....conftest import AdaptiveInputInteraction, _correlated, _summed, _weighted


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


def _pushed_gas(n=500):
    """Return a group pushed by a constant force, whose velocity change depends on the mass."""
    return CompositeSystem(
        elements=[GroupElement('gas', element=MassPointElement, size=n)],
        interactions=[ConstantForceInteraction('push', target='gas', F=[1.0, 0.0])],
        space=ContinuousSpace(n_dim=2),
    )


class FrameCounterObserver(BaseObserver):
    """Observer that keeps the times of the frames it receives."""

    def _prepare(self, context):
        """Start with no frames."""
        self.times_ = []
        self.stopped_ = False

    def _run(self, inputs):
        """Keep the time of the frame."""
        if inputs is not None:
            self.times_.append(inputs.time)

    def _release(self):
        """Mark the end of the run."""
        self.stopped_ = True


def _growth(r=1.0):
    """Return a system of exponential growth."""
    return CompositeSystem(
        elements=[PopulationElement('a', x=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=r)],
    )


def _epidemic():
    """Return a stochastic SIR system."""
    return CompositeSystem(
        elements=[AmountElement('S', x=95), AmountElement('I', x=5), AmountElement('R')],
        interactions=[
            InfectionInteraction('infection', susceptible='S', infected='I', beta=0.01, stochastic=True),
            RecoveryInteraction('recovery', infected='I', recovered='R', gamma=0.3, stochastic=True),
        ],
    )


def _logistic():
    """Return a logistic population."""
    return CompositeSystem(
        elements=[PopulationElement('rabbits', x=10)],
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

    def save(self, agent, system):
        """Add the rate to the savings."""
        del system
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
    106.84701129798354,
    108.17826266273447,
    105.65734164064845,
    110.49449907498547,
    109.94445714673115,
    110.82007444147138,
    114.5517667447161,
    111.32038165744878,
    109.67041359152822,
    108.34081429617326,
]


def test_simulation_updates_one_per_entry_weight_live():
    """Test that a live update of one component of a per-entry parameter changes that entry only."""
    schedule = {'sum.weights[1]': lambda t: 1.0 if t < 1 else 10.0}
    trajectory = simulate(_summed(['a', 'b', 'c']), until=2, dt=1, schedule=schedule)
    assert trajectory.values('target.x').tolist() == [0.0, 7.0, 25.0]


def test_closed_economy_changes_money_only_by_lending_and_repayment():
    """Test that total money changes by lending minus repayment, and every transaction-flow row sums to zero."""
    trajectory = simulate(
        StockFlowConsistentEconomySystem().prepare().system_,
        until=100,
        dt=1,
        observers=[ConservationObserver()],
    )
    money = (
        trajectory.values('households.deposits')
        + trajectory.values('firms.deposits')
        + trajectory.values('bank.deposits')
    )
    created = trajectory.values('circuit.lending') - trajectory.values('circuit.repayments')
    assert np.diff(money) == pytest.approx(created[:-1], abs=1e-9)
    for frame in (0, 50, 100):
        matrix = loops(trajectory, kind='flow-matrix', frame=frame).matrix
        assert matrix.sum(axis=1) == pytest.approx(0.0, abs=1e-12)


def test_trajectory_returns_group_arrays():
    """Test that a group state over 100 frames has the shape `(100, 500, 2)`."""
    trajectory = simulate(_pushed_gas(), until=0.99, dt=0.01)
    assert trajectory.values('gas.x').shape == (100, 500, 2)
    assert trajectory.values('gas.m').shape == (100, 500)


def test_group_parameter_changes_by_schedule():
    """Test that a scheduled mass of every member applies to the frames after the change."""
    schedule = {'gas.m': lambda t: 1.0 if t < 0.2 - 1e-9 else 2.0}
    trajectory = simulate(_pushed_gas(), until=0.4, dt=0.01, schedule=schedule)
    changed = int(np.argmax(np.all(trajectory.values('gas.m') == 2.0, axis=1)))
    assert changed > 0
    accelerations = np.diff(trajectory.values('gas.x_d1')[:, :, 0], axis=0) / 0.01
    assert np.allclose(accelerations[changed:], 0.5)
    assert np.allclose(accelerations[: changed - 1], 1.0)


def test_group_parameter_takes_per_member_values():
    """Test that a scheduled group parameter accepts one value per member."""
    trajectory = simulate(_pushed_gas(3), until=0.1, dt=0.1, schedule={'gas.m': lambda t: [1.0, 2.0, 4.0]})
    assert trajectory.values('gas.m')[-1].tolist() == [1.0, 2.0, 4.0]


def test_group_parameter_with_wrong_shape_raises_error():
    """Test that a scheduled group parameter rejects values that do not broadcast to the group."""
    with pytest.raises(ValidationError, match=r'At t = 0\.0, the schedule sets gas\.m .* broadcasts'):
        simulate(_pushed_gas(3), until=0.1, dt=0.1, schedule={'gas.m': lambda t: [1.0, 2.0]})


def test_reflecting_box_keeps_members_inside_ode_run():
    """Test that no member crosses a wall of a reflecting box in an ODE system."""
    trajectory = simulate(IdealGasSystem(500, kT=4.0, random_state=1).prepare().system_, until=2, dt=0.01)
    positions = trajectory.values('gas.x')
    assert np.all((positions >= 0) & (positions <= 1))


@pytest.mark.benchmark
def test_ideal_gas_runs_in_real_time():
    """Test that an ideal gas of 10000 molecules keeps a real-time clock within 0.1 seconds."""
    simulation = Simulation(
        IdealGasSystem(10_000, random_state=0).prepare().system_,
        dt=0.01,
        clock=RealTimeClock(speed=1.0),
        record=False,
    ).prepare()
    simulation.run(until=5)
    assert simulation.clock_.max_lag_ <= 0.1


def test_simulation_hands_frames_to_observers():
    """Test that observers receive one frame per dt and the end of the run."""
    counter = FrameCounterObserver()
    Simulation(_growth(), dt=0.1, observers=[counter]).prepare().run(until=1)
    assert np.allclose(counter.times_, np.linspace(0, 1, 11))
    assert counter.stopped_


def test_simulation_discrete_frames_are_integer_times():
    """Test that discrete maps produce frames at integer times."""
    system = CompositeSystem(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')])
    trajectory = simulate(system, until=10)
    assert trajectory.times.tolist() == list(map(float, range(11)))


def test_simulation_discrete_noninteger_dt_raises_error():
    """Test that a discrete map needs an integer dt."""
    system = CompositeSystem(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')])
    with pytest.raises(ValidationError, match='positive integer'):
        Simulation(system, dt=0.5).prepare().run(until=1)


def test_simulation_wrong_solver_raises_error():
    """Test that a solver of another form is rejected."""
    with pytest.raises(SolverError, match='GillespieSolver'):
        Simulation(_growth(), solver=GillespieSolver()).prepare().run(until=1)


def test_simulation_real_time_paces_frames():
    """Test that a real-time run takes at least its simulated duration."""
    clock = RealTimeClock(speed=1.0)
    start = time.perf_counter()
    simulate(_growth(), until=0.5, dt=0.1, clock=clock)
    assert time.perf_counter() - start >= 0.5
    assert clock.max_lag_ >= 0.0


def test_simulation_schedule_unknown_parameter_raises_error():
    """Test that only parameters of the problem can be scheduled."""
    with pytest.raises(ParameterError, match=r"'g\.k'"):
        simulate(_growth(), until=0.1, schedule={'g.k': lambda t: 1.0})


def test_simulation_schedule_invalid_value_raises_error():
    """Test that scheduled values are checked against their domain, with the parameter and the time named."""
    system = CompositeSystem(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')])
    schedule = {'c.b': lambda t: 1.0 if t < 3 else -1.0}
    message = r'At t = 3\.0, the schedule sets c\.b = -1\.0, which is not a value .* in the domain \(0, \+oo\)'
    with pytest.raises(ValidationError, match=message):
        simulate(system, until=5, schedule=schedule)


def test_parameter_error_is_a_validation_error():
    """Test that one except clause of `ValidationError` catches an unknown scheduled parameter too."""
    with pytest.raises(ValidationError):
        simulate(_growth(), until=0.1, schedule={'g.k': lambda t: 1.0})


def test_simulation_trajectory_before_run_raises_error():
    """Test that the trajectory needs a run first."""
    with pytest.raises(SimulationError, match='has not been run'):
        _ = Simulation(_growth()).trajectory_


def test_simulation_without_recording_has_no_trajectory():
    """Test that disabling recording removes the trajectory."""
    simulation = Simulation(_growth(), record=False).prepare().run(until=0.1)
    with pytest.raises(SimulationError, match='record=True'):
        _ = simulation.trajectory_


def test_trajectory_unknown_name_raises_error():
    """Test that a trajectory names the available variables."""
    trajectory = simulate(_growth(), until=0.1)
    with pytest.raises(KeyError, match=r'a\.x, g\.r'):
        trajectory.values('b.x')


def test_simulation_same_random_state_is_reproducible():
    """Test that a fixed random state reproduces a stochastic run."""
    first = simulate(_epidemic(), until=20, dt=0.5, random_state=0)
    second = simulate(_epidemic(), until=20, dt=0.5, random_state=0)
    assert np.array_equal(first.states, second.states)


def test_simulation_agent_parameters_follow_the_schedule():
    """Test that a scheduled agent group parameter is read by the rules of the run."""
    system = CompositeSystem(elements=[GroupElement('savers', element=SaverElement, size=3)])
    trajectory = simulate(system, until=2, schedule={'savers.rate': lambda t: 5.0})
    assert trajectory.get_agents('savers', 2).values['savings'].tolist() == [10.0, 10.0, 10.0]


def test_simulation_extend_runs_in_two_parts():
    """Test that an extended run appends frames that follow a scheduled parameter change."""
    schedule = {'growth.r': lambda t: 1.0 if t < 5 else 0.0}
    simulation = Simulation(_logistic(), dt=1.0, schedule=schedule).prepare().run(until=5)
    trajectory = simulation.extend(until=10).trajectory_
    reference = simulate(_logistic(), until=5, dt=1.0).values('rabbits.x')
    assert trajectory.times.tolist() == [float(time) for time in range(11)]
    assert trajectory.values('rabbits.x')[:6] == pytest.approx(reference)
    assert np.all(trajectory.values('rabbits.x')[5:] == trajectory.values('rabbits.x')[5])


def test_simulation_extend_backwards_raises_error():
    """Test that extending to an earlier time raises a named error that names the current time."""
    simulation = Simulation(_logistic(), dt=1.0).prepare().run(until=5)
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
    simulation = Simulation(_growth(r=0.0), observers=[counter]).prepare().run(until=until)
    assert len(counter.times_) == frames
    assert simulation.dt_ == dt


def test_trajectory_lists_its_names():
    """Test that a trajectory lists the names of its states, parameters, reporters, and observables."""
    system = HilsaFisherySystem().prepare().system_
    observables = HilsaFisherySystem().get_run_settings()['observables']
    trajectory = simulate(system, until=2, dt=1, observables=observables)
    names = trajectory.names
    assert 'hilsa.adult_weight' in names
    assert all(name in names for name in trajectory.problem.states.names)


def test_trajectory_gives_agent_locations_over_frames():
    """Test that the locations of a group of agents come as one array per frame."""
    trajectory = simulate(BoidsSystem().prepare().system_, until=3, random_state=0)
    locations = trajectory.agent_locations('flock')
    assert len(locations) == len(trajectory.times)
    np.testing.assert_allclose(locations[-1], trajectory.get_agents('flock', len(trajectory.times) - 1).locations)


def test_initial_state_replaces_the_values_of_components():
    """Test that a simulation starts from given initial values by state name and rejects unknown names."""
    system = CompositeSystem(
        elements=[PopulationElement('rabbits', x=10)],
        interactions=[ExponentialGrowthInteraction('growth', target='rabbits', r=0.0)],
    )
    trajectory = simulate(system, until=1.0, dt=1.0, initial_state={'rabbits.x': 20.0})
    assert trajectory.values('rabbits.x')[0] == pytest.approx(20.0)
    with pytest.raises(ValidationError, match=r'foxes\.x'):
        Simulation(system, dt=1.0, initial_state={'foxes.x': 1.0}).prepare().run(until=1.0)


def test_simulation_schedules_a_bounded_control():
    """Test that a scheduled control takes its value at every frame, which laws read and trajectories record."""
    schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
    trajectory = simulate(CompositeSystem(elements=[HeaterElement('heater')]), until=3.0, dt=0.5, schedule=schedule)
    np.testing.assert_array_equal(trajectory.values('heater.power'), [0, 0, 1, 1, 1, 1, 1])
    assert trajectory.values('heater.T')[2] == pytest.approx(0.0)
    assert trajectory.values('heater.T')[-1] == pytest.approx(1 - np.exp(-2.0), rel=1e-3)


def test_correlated_noise_repeats_with_same_seed():
    """Test that two runs of a correlated `WienerProcessSpec` with random state 0 on the Numba backend are equal."""
    system = _correlated([[1.0, 0.5], [0.5, 1.0]])
    runs = [simulate(system, until=2, backend=NumbaBackend(), random_state=0, dt=0.1).states for _ in range(2)]
    np.testing.assert_array_equal(*runs)


def test_perfect_correlation_keeps_states_equal():
    """Test that processes of correlation 1 keep two states with unit noise equal at every frame."""
    trajectory = simulate(_correlated([[1.0, 1.0], [1.0, 1.0]]), until=2, random_state=0, dt=0.1)
    np.testing.assert_allclose(trajectory.values('a.n'), trajectory.values('b.n'), atol=1e-12)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_unnamed_noise_reproduces_stored_trajectory(backend):
    """Test that geometric Brownian motion with random state 0 repeats the stored trajectory on both backends."""
    simulation = simulate(
        GeometricBrownianMotionSystem().prepare().system_,
        until=0.5,
        backend=backend(),
        random_state=0,
        dt=0.05,
    )
    trajectory = simulation
    np.testing.assert_array_equal(trajectory.values('asset.price'), STORED_GEOMETRIC_BROWNIAN_MOTION)


class _ReleaseCountingSolver(RK4Solver):
    """RK4 solver that counts the final runs that release it."""

    def _prepare(self, context):
        """Fit the solver and start with no release."""
        super()._prepare(context)
        self.releases_ = 0

    def _release(self):
        """Count the release."""
        self.releases_ += 1


def test_simulation_releases_the_solver_at_the_last_frame():
    """Test that a run fits the solver once and releases it with the run of its last frame."""
    simulation = Simulation(_growth(), solver=_ReleaseCountingSolver(), dt=0.1).prepare().run(until=1)
    assert simulation.stepper_.releases_ == 1
    assert simulation.trajectory_.times.size == 11


def test_simulation_prepare_compiles_and_chooses_the_solver_without_running():
    """Test that preparing a simulation takes no input, compiles the system, and runs nothing."""
    simulation = Simulation(_growth()).prepare()
    assert simulation.problem_.states.names == ['a.x']
    assert simulation.stepper_ is not None
    assert not hasattr(simulation, 'time_')
    with pytest.raises(SimulationError, match='not run'):
        simulation.extend(until=1)


def test_simulation_run_takes_the_end_time():
    """Test that running a simulation takes one input, the time to stop at, and returns the simulation."""
    assert list(inspect.signature(Simulation.prepare).parameters) == ['self', 'context']
    assert list(inspect.signature(Simulation.run).parameters) == ['self', 'until', 'final']
    simulation = Simulation(_growth(), dt=1.0).prepare()
    with pytest.raises(ValidationError, match='until'):
        simulation.run()
    assert simulation.run(2) is simulation
    assert simulation.time_ == 2.0


def test_schedule_shaped_parameter_of_wrong_shape_raises_error():
    """Test that a matrix parameter scheduled with the transposed shape names the parameter and both shapes."""
    schedule = {'input.W': lambda t: np.ones((2, 4))}
    with pytest.raises(ValidationError, match=r'(?s)input\.W = .* of shape \(4, 2\).* has shape \(2, 4\)'):
        simulate(_weighted(np.ones((4, 2)), [1.0, 2.0]), until=0.1, dt=0.1, schedule=schedule)


def test_trajectory_values_shaped_parameter_has_its_shape():
    """Test that a matrix parameter is read from a trajectory with its shape after a change as a whole."""
    weights = np.arange(8.0).reshape(4, 2)
    schedule = {'input.W': lambda t: weights if t < 0.2 else 2 * weights}
    values = simulate(_weighted(weights, [1.0, 2.0]), until=0.3, dt=0.1, schedule=schedule).values('input.W')
    assert values.shape[1:] == (4, 2)
    np.testing.assert_array_equal(values[-1], 2 * weights)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_trajectory_values_adaptive_shaped_parameter_of_an_interaction(backend):
    """Test that an adaptive matrix parameter of an interaction is recorded with its shape at every frame."""
    weights = np.arange(8.0).reshape(4, 2)
    system = _weighted(weights, [1.0, 2.0], interaction=AdaptiveInputInteraction)
    trajectory = simulate(system, until=1.0, dt=0.1, backend=backend())
    assert trajectory.values('input.W').shape == (11, 4, 2)
    np.testing.assert_allclose(trajectory.values('hidden.x')[-1], weights @ [1.0, 2.0], atol=1e-12)


HEALTH = FiniteEnumeratedSet(['susceptible', 'infected', 'recovered'])


class PersonElement(Element):
    """Person whose health is one of three labels."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        health: Annotated[object, StateSpec(domain=HEALTH)] = 'susceptible',
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health


class ProgressionInteraction(Interaction):
    """Progression of the health of its target to the next label."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        step: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.step = step

    def _get_laws(self, s):
        """Return the progression of the health."""
        return [UpdateLaw(s.target.health, s.target.health + s.step)]


def test_checked_mode_catches_values_outside_alphabet():
    """Test that checked mode names the state when a law leaves the alphabet."""
    system = CompositeSystem(
        elements=[PersonElement('ann', health='recovered')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    with pytest.raises(ValidationError, match=r'ann\.health took the values 3\.0 at t = 1\.0, outside its domain'):
        simulate(system, until=1, checked=True)
    assert simulate(system, until=1).values('ann.health').shape == (2,)


@pytest.mark.benchmark
def test_large_game_of_life_runs_in_real_time(tmp_path):
    """Test that a 1000 x 1000 Game of Life on Numba sustains 30 steps per second while streaming images."""
    observer = RerunObserver('life', save_path=tmp_path / 'life.rrd')
    simulation = Simulation(
        GameOfLifeSystem((1000, 1000)).prepare().system_,
        dt=1,
        backend=NumbaBackend(),
        record=False,
        observers=[observer],
    ).prepare()
    simulation.run(until=2)
    simulation.observers = [RerunObserver('life', save_path=tmp_path / 'paced.rrd')]
    simulation.clock = RealTimeClock(30.0)
    start = time.perf_counter()
    simulation.run(until=90)
    assert 90 / (time.perf_counter() - start) >= 29.0
    assert simulation.problem_.lattice[0].fused
    assert simulation.observers[0].logged_frames_ == 91


class ChainElement(Element):
    """Chain of two symbols whose first symbol stays with a probability."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        stay: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.stay = stay

    def _get_laws(self, s):
        """Return the transition of the chain."""
        return [TransitionLaw(s.x, [[s.stay, 1 - s.stay], [0.4, 0.6]])]


# Default and given solvers of a simulation, moved from the tests of `skcomplex.core.engine.solvers._defaults`.


def test_simulation_without_a_solver_runs_the_default_solver():
    """Test that a simulation without a solver runs the default solver of the system's form."""
    system = CompositeSystem(
        elements=[PopulationElement('a', x=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    simulation = Simulation(system).prepare().run(until=0.1)
    assert type(simulation.stepper_) is DEFAULT_SOLVERS[simulation.problem_.form] is ScipyODESolver


def test_simulation_runs_the_solver_instance_passed():
    """Test that a simulation runs the solver instance it is given."""
    system = CompositeSystem(
        elements=[PopulationElement('a', x=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    solver = ScipyODESolver(method='RK45')
    assert Simulation(system, solver=solver).prepare().run(until=0.1).stepper_ is solver
