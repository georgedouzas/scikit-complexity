"""Test the `_simulate` module of `skcomplex.core.engine.runtime`."""

from collections.abc import Mapping
from typing import Annotated, ClassVar

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import JumpInteraction, RateInteraction
from skcomplex.core.components.spaces import LatticeSpace, StaticTime
from skcomplex.core.components.system import BaseSystem, System
from skcomplex.core.engine.backends import JaxBackend
from skcomplex.core.engine.runtime import RecorderObserver, Simulation, Trajectory, simulate
from skcomplex.core.engine.runtime._simulate import _choose_solver
from skcomplex.core.engine.solvers import GillespieSolver, MonteCarloSolver, RootSolver, ScipyODESolver
from skcomplex.core.errors import TagError, ValidationError
from skcomplex.core.specs import ObservableSpec, StateSpec, Tags


class TankElement(Element):
    """Element with one real state."""

    def __init__(self, label: str, x: Annotated[float, StateSpec()] = 1.0) -> None:
        super().__init__(label)
        self.x = x


def _double(namespace):
    """Return twice the state of a tank."""
    return 2 * namespace.x


class DecaySystem(BaseSystem):
    """System of exponential decay, with tuned run settings."""

    until = 2.0
    dt = 0.5
    initial_state: ClassVar[Mapping[str, object]] = {'tank.x': 4.0}
    observables = (ObservableSpec('double', _double, element='tank'),)

    def __init__(self, label: str = 'decay', k: float = 1.0) -> None:
        super().__init__(label)
        self.k = k

    def _build_system(self) -> System:
        decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -self.k * s.target.x)
        return System(self.label, elements=[TankElement('tank')], interactions=[decay])


def _jump_system() -> System:
    """Return a system of one jump interaction."""
    jump = JumpInteraction('drop', target='tank', changes={'x': -1.0}, rate=lambda s: s.target.x)
    return System(elements=[TankElement('tank', x=5.0)], interactions=[jump])


def test_simulate_returns_the_trajectory_until_a_time():
    """Test that simulating a composite system returns its trajectory until the end time."""
    decay = RateInteraction('decay', target='tank', state='x', rate=lambda s: -s.target.x)
    trajectory = simulate(System(elements=[TankElement('tank')], interactions=[decay]), until=1.0)
    assert isinstance(trajectory, Trajectory)
    assert trajectory.times[-1] == pytest.approx(1.0)
    assert trajectory.values('tank.x')[-1] == pytest.approx(np.exp(-1.0), rel=1e-3)


def test_simulate_reads_the_tuned_settings_of_the_class():
    """Test that the end time, the time between frames, the initial state, and the observables come from the class."""
    trajectory = simulate(DecaySystem(k=0.5))
    assert trajectory.times.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert trajectory.values('tank.x')[0] == 4.0
    assert trajectory.values('tank.double')[-1] == pytest.approx(8.0 * np.exp(-1.0), rel=1e-3)
    assert simulate(DecaySystem(), until=1.0, initial_state={'tank.x': 1.0}).values('tank.x')[0] == 1.0


def test_simulate_takes_the_time_between_frames_and_observers():
    """Test that a time between frames replaces the tuned one, and that observers receive every frame."""
    recorder = RecorderObserver()
    trajectory = simulate(DecaySystem(), dt=1.0, observers=[recorder])
    assert trajectory.times.tolist() == [0.0, 1.0, 2.0]
    assert recorder.trajectory_.times.tolist() == [0.0, 1.0, 2.0]


def test_simulate_requires_an_end_time():
    """Test that a system without a tuned end time needs `until`."""
    with pytest.raises(ValidationError, match=r'has no tuned end time\. Pass `until`'):
        simulate(System(elements=[TankElement('tank')]))


def test_simulate_rejects_what_is_no_system():
    """Test that simulating a name raises an error."""
    with pytest.raises(ValidationError, match='simulate takes a system'):
        simulate('sir', until=1.0)


def test_simulate_rejects_a_solver_of_other_law_kinds_before_compiling(monkeypatch):
    """Test that an ODE solver on a jump system raises an error naming the solver and the law kind before compiling."""

    def fail(*args, **kwargs):
        raise AssertionError

    monkeypatch.setattr(Simulation, 'prepare', fail)
    with pytest.raises(TagError, match='The solver ScipyODESolver does not support the laws jump'):
        simulate(_jump_system(), until=1.0, solver=ScipyODESolver())


def test_simulate_rejects_a_backend_of_other_spaces():
    """Test that a backend that does not support the space of a system is rejected, naming the space."""
    group = GroupElement('cells', element=TankElement, space=LatticeSpace((3,)))
    decay = RateInteraction('decay', target='cells', state='x', rate=lambda s: -s.target.x)
    with pytest.raises(TagError, match='The backend JaxBackend does not support the space lattice'):
        simulate(System(elements=[group], interactions=[decay]), until=1.0, backend=JaxBackend())


def test_simulate_runs_a_solver_that_supports_the_tags():
    """Test that a jump system runs with a jump solver."""
    trajectory = simulate(_jump_system(), until=100.0, solver=GillespieSolver(), random_state=0)
    assert trajectory.values('tank.x')[-1] == 0.0


def test_tags_choose_the_default_solver_when_one_default_supports_them():
    """Test that tags that fit one default solver choose it, and that other tags leave the choice to the form."""
    static = Tags(time=frozenset({'static'}), laws=frozenset({'equation'}))
    sampling = Tags(time=frozenset({'discrete'}), space=frozenset({'lattice'}), laws=frozenset({'sampling'}))
    assert isinstance(_choose_solver(static), RootSolver)
    assert isinstance(_choose_solver(sampling), MonteCarloSolver)
    assert _choose_solver(Tags(time=frozenset({'continuous'}), laws=frozenset({'rate'}))) is None
    assert _choose_solver(Tags(laws=None)) is None
    assert StaticTime.time_tag == 'static'
