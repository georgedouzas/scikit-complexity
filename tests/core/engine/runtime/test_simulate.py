"""Test the `_simulate` module of `skcomplex.core.engine.runtime`."""

from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.components import Observable, Tags
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import normal
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.backends import JaxBackend
from skcomplex.core.engine.runtime import RecorderObserver, Trajectory, _simulate
from skcomplex.core.engine.runtime._simulate import _choose_solver, run_problem
from skcomplex.core.engine.solvers import GillespieSolver, MonteCarloSolver, RootSolver
from skcomplex.core.errors import ConservationError, TagError, ValidationError
from skcomplex.core.systems import BaseSystem, System


def _decay(label: str = 'decay', target: str = 'tank', k: float = 1.0) -> Interaction:
    """Return an interaction that makes the level of a tank decay at the rate `k`."""
    return Interaction(label, roles={'target': target}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), -k * s.target.x)])


def _double(namespace):
    """Return twice the state of a tank."""
    return 2 * namespace.x


class DecaySystem(BaseSystem):
    """System of exponential decay, with tuned run settings."""

    until = 2.0
    dt = 0.5
    initial_state: ClassVar[Mapping[str, object]] = {'tank.x': 4.0}
    observables = (Observable('double', _double, element='tank'),)

    def __init__(self, label: str = 'decay', k: float = 1.0) -> None:
        super().__init__(label)
        self.k = k

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of one decaying tank."""
        super()._prepare(context)
        tank = Element('tank', values={'x': 1.0})
        self.system_ = System(self.label, elements=[tank], interactions=[_decay(k=self.k)])


def _level(namespace):
    """Return the level of a tank."""
    return namespace.x


class LeakySystem(BaseSystem):
    """System of one leaky tank, with tuned invariants and schedule."""

    until = 2.0
    dt = 0.5
    observables = (Observable('level', _level, element='tank'),)
    invariants: ClassVar[Mapping[str, float]] = {'tank.level': 1e-9}
    schedule: ClassVar[Mapping[str, object]] = {'tank.k': lambda t: 0.0}

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of one tank that leaks at the rate of its value `k`."""
        super()._prepare(context)
        leak = Interaction(
            'leak',
            roles={'target': 'tank'},
            laws=lambda s: [RateLaw(s.target.x.diff(s.t), -s.target.k * s.target.x)],
        )
        self.system_ = System(self.label, elements=[Element('tank', values={'x': 1.0, 'k': 1.0})], interactions=[leak])


def _jump_system(solver: GillespieSolver | str = 'auto') -> System:
    """Return a system of one jump interaction."""
    jump = Interaction('drop', roles={'target': 'tank'}, laws=lambda s: [JumpLaw(s.target.x, {s.target.x: -1.0})])
    return System(elements=[Element('tank', values={'x': 5.0})], interactions=[jump], solver=solver)


def test_simulate_returns_the_trajectory_until_a_time():
    """Test that simulating a system returns its trajectory until the end time."""
    system = System(elements=[Element('tank', values={'x': 1.0})], interactions=[_decay()]).prepare()
    trajectory = system.simulate(until=1.0)
    assert isinstance(trajectory, Trajectory)
    assert trajectory.times[-1] == pytest.approx(1.0)
    assert trajectory.get_values('tank.x')[-1] == pytest.approx(np.exp(-1.0), rel=1e-3)


def test_simulate_reads_the_tuned_settings_of_the_class():
    """Test that the end time, the time between frames, the initial state, and the observables come from the class."""
    trajectory = DecaySystem(k=0.5).prepare().simulate()
    assert trajectory.times.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert trajectory.get_values('tank.x')[0] == 4.0
    assert trajectory.get_values('tank.double')[-1] == pytest.approx(8.0 * np.exp(-1.0), rel=1e-3)
    assert DecaySystem().prepare().simulate(until=1.0, initial_state={'tank.x': 1.0}).get_values('tank.x')[0] == 1.0


def test_simulate_takes_the_time_between_frames_and_observers():
    """Test that a time between frames replaces the tuned one, and that observers receive every frame."""
    recorder = RecorderObserver()
    trajectory = DecaySystem().prepare().simulate(dt=1.0, observers=[recorder])
    assert trajectory.times.tolist() == [0.0, 1.0, 2.0]
    assert recorder.trajectory_.times.tolist() == [0.0, 1.0, 2.0]


def test_simulate_requires_an_end_time():
    """Test that a system without a tuned end time needs `until`."""
    system = System(elements=[Element('tank', values={'x': 1.0})], interactions=[_decay()]).prepare()
    with pytest.raises(ValidationError, match=r'has no tuned end time\. Pass `until`'):
        system.simulate()


def test_run_problem_rejects_what_is_no_system():
    """Test that running a name raises an error that names what it took."""
    with pytest.raises(ValidationError, match=r"simulate takes a system, not 'sir'"):
        run_problem('sir', None, until=1.0)


def test_prepare_rejects_a_backend_of_other_spaces():
    """Test that a backend that does not support the space of a system is rejected, naming the space."""
    cells = Element('cells', values={'x': 1.0}, space=LatticeSpace((3,)))
    system = System(elements=[cells], interactions=[_decay(target='cells')], backend=JaxBackend())
    with pytest.raises(TagError, match='The backend JaxBackend does not support the space lattice'):
        system.prepare()


def test_simulate_runs_a_solver_that_supports_the_tags():
    """Test that a jump system runs with a jump solver."""
    trajectory = _jump_system(GillespieSolver()).prepare().simulate(until=100.0, random_state=0)
    assert trajectory.get_values('tank.x')[-1] == 0.0


def test_tags_choose_the_default_solver_when_one_default_supports_them():
    """Test that tags that fit one default solver choose it, and that other tags leave the choice to the form."""
    static = Tags(time=frozenset({'static'}), laws=frozenset({'equation'}))
    sampling = Tags(time=frozenset({'discrete'}), space=frozenset({'lattice'}), laws=frozenset({'sampling'}))
    assert isinstance(_choose_solver(static), RootSolver)
    assert isinstance(_choose_solver(sampling), MonteCarloSolver)
    assert _choose_solver(Tags(time=frozenset({'continuous'}), laws=frozenset({'rate'}))) is None
    assert _choose_solver(Tags(laws=None)) is None


def test_simulate_reads_the_tuned_invariants_and_schedule():
    """Test that the tuned schedule holds the level, which the tuned invariant checks."""
    system = LeakySystem().prepare()
    assert system.simulate().get_values('tank.x').tolist() == [1.0] * 5
    with pytest.raises(ConservationError, match=r'invariant tank\.level'):
        system.simulate(schedule={'tank.k': lambda t: 1.0})
    assert system.simulate(schedule={'tank.k': lambda t: 1.0}, invariants={}).get_values('tank.x')[-1] < 1.0


def test_prepare_rejects_a_tuned_backend_of_other_spaces():
    """Test that a backend of a ready-made system that does not support its space is rejected, naming the space."""

    class CellsSystem(BaseSystem):
        """System of decaying cells on a lattice."""

        def _prepare(self, context: Sequence[Observable] | None = None) -> None:
            """Set the system of the cells."""
            super()._prepare(context)
            cells = Element('cells', values={'x': 1.0}, space=LatticeSpace((3,)))
            self.system_ = System(self.label, elements=[cells], interactions=[_decay(target='cells')])

    with pytest.raises(TagError, match='The backend JaxBackend does not support the space lattice'):
        CellsSystem(backend=JaxBackend()).prepare()


def test_simulate_leaves_the_solver_of_nested_systems_to_the_compiled_form(monkeypatch):
    """Test that a system of nested systems runs without the tags choosing a solver for the whole."""

    def fail(tags):
        raise AssertionError

    monkeypatch.setattr(_simulate, '_choose_solver', fail)
    left = System('left', elements=[Element('tank', values={'x': 1.0})], interactions=[_decay()])
    trajectory = System(systems=[left]).prepare().simulate(until=1.0)
    assert trajectory.get_values('left/tank.x')[-1] == pytest.approx(np.exp(-1.0), rel=1e-3)


def _kicked(deviation: float, label: str = 'walker') -> System:
    """Return a system of one element whose state takes a normal kick of a deviation at every step."""
    kick = Interaction(
        'kick',
        roles={'target': label},
        laws=lambda s: [UpdateLaw(s.target.x, s.target.x + normal(f'{label}_kick', 0, s.target.s))],
    )
    return System(elements=[Element(label, values={'x': 0.0, 's': deviation})], interactions=[kick]).prepare()


def test_simulate_records_the_draws_of_a_law():
    """Test that a noisy map run with `record_draws=True` holds 100 kicks, each with its deviation."""
    trajectory = _kicked(0.5).simulate(until=100, random_state=0, record_draws=True)
    record = trajectory.draws.draws['kick.walker_kick']
    assert record.values.size == 100
    assert record.steps.tolist() == list(range(1, 101))
    assert np.all(record.arguments['deviation'] == 0.5)
    np.testing.assert_allclose(np.cumsum(record.values), trajectory.get_values('walker.x')[1:])
    assert trajectory.log_probability == pytest.approx(stats.norm.logpdf(record.values, 0, 0.5).sum())


def test_simulate_without_recording_holds_no_draws():
    """Test that a run that neither records nor replays holds no draws."""
    trajectory = _kicked(0.5).simulate(until=3, random_state=0)
    assert trajectory.draws is None
    assert trajectory.log_probability is None


def test_replay_forces_the_recorded_draws_and_scores_them():
    """Test that a replay keeps the recorded states and scores the draws under its own parameters."""
    recorded = _kicked(0.5).simulate(until=50, random_state=0, record_draws=True)
    same = _kicked(0.5).simulate(until=50, random_state=0, replay=recorded)
    other = _kicked(1.0).simulate(until=50, random_state=7, replay=recorded)
    values = recorded.draws.draws['kick.walker_kick'].values
    np.testing.assert_allclose(same.get_values('walker.x'), recorded.get_values('walker.x'))
    np.testing.assert_allclose(other.get_values('walker.x'), recorded.get_values('walker.x'))
    assert same.log_probability == pytest.approx(recorded.log_probability)
    assert other.log_probability == pytest.approx(stats.norm.logpdf(values, 0, 1.0).sum())


def test_replay_rejects_a_system_whose_draws_have_other_names():
    """Test that a replay names the draws the system makes and the trajectory lacks."""
    recorded = _kicked(0.5).simulate(until=5, random_state=0, record_draws=True)
    with pytest.raises(ValidationError, match=r"draws 'kick\.hopper_kick', which the replayed trajectory lacks"):
        _kicked(0.5, label='hopper').simulate(until=5, random_state=0, replay=recorded)


def test_replay_needs_a_trajectory_with_draws():
    """Test that replaying a trajectory recorded without its draws raises a named error."""
    recorded = _kicked(0.5).simulate(until=5, random_state=0)
    with pytest.raises(ValidationError, match='holds no draws'):
        _kicked(0.5).simulate(until=5, random_state=0, replay=recorded)
