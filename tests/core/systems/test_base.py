"""Test the `_base` module of `skcomplex.core.systems`."""

from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from sklearn.base import clone

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, ObjectiveLaw, UpdateLaw
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.backends import BaseBackend, NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.engine.solvers import BaseSolver, RK4Solver
from skcomplex.core.errors import NotPreparedError
from skcomplex.core.systems import BaseSystem, System
from skcomplex.core.types import Role
from skcomplex.systems.biology.ecology import LotkaVolterraSystem


class MeadowSystem(BaseSystem):
    """System of one Lotka-Volterra meadow whose prey grow at the rate `r`."""

    until = 4.0

    def __init__(
        self,
        label: str = 'system',
        r: float = 1.1,
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.r = r

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the meadow."""
        super()._prepare(context)
        self.system_ = System(self.label, systems=[LotkaVolterraSystem(label='meadow', r=self.r)])


def test_simulate_returns_trajectory() -> None:
    """Simulating a prepared system returns a trajectory up to the end time."""
    trajectory = MeadowSystem().prepare().simulate(until=2.0)
    assert isinstance(trajectory, Trajectory)
    assert trajectory.times[-1] == pytest.approx(2.0)


def test_simulate_uses_tuned_end_time() -> None:
    """A run without an end time uses the tuned one."""
    assert MeadowSystem().prepare().simulate().times[-1] == pytest.approx(4.0)


def test_simulate_runs_many_times() -> None:
    """One prepared system simulates twice with different end times and initial states."""
    system = MeadowSystem().prepare()
    short = system.simulate(until=1.0)
    other = system.simulate(until=2.0, initial_state={'meadow/prey.x': 10.0})
    assert short.times[-1] == pytest.approx(1.0)
    assert other.get_values('meadow/prey.x')[0] == pytest.approx(10.0)


def test_simulate_before_prepare_raises() -> None:
    """Simulating a system that was never prepared raises an error that asks for prepare."""
    with pytest.raises(NotPreparedError, match='Call `prepare\\(\\)` first'):
        MeadowSystem().simulate(until=1.0)


def test_simulate_after_set_params_raises() -> None:
    """Simulating after a parameter changed raises an error that names the parameter."""
    system = MeadowSystem().prepare().set_params(r=1.5)
    with pytest.raises(NotPreparedError, match=r'parameters r of the system'):
        system.simulate(until=1.0)


def test_prepare_reaches_run() -> None:
    """A model parameter set before prepare reaches the components of the run."""
    slow = MeadowSystem(r=0.1).prepare().simulate(until=2.0).get_values('meadow/prey.x')[-1]
    fast = MeadowSystem().set_params(r=2.0).prepare().simulate(until=2.0).get_values('meadow/prey.x')[-1]
    assert fast > slow


def test_clone_keeps_model_parameters() -> None:
    """Cloning a system keeps its model parameters."""
    assert clone(MeadowSystem(r=0.7)).get_params() == {'backend': 'auto', 'label': 'system', 'r': 0.7, 'solver': 'auto'}


def test_get_params_lists_the_solver_and_the_backend() -> None:
    """A ready-made system lists the solver and the backend among its parameters, and a clone keeps them."""
    system = MeadowSystem(r=0.7, solver=RK4Solver(dt=0.01), backend=NumbaBackend())
    params = system.get_params(deep=False)
    assert isinstance(params['solver'], RK4Solver)
    assert isinstance(params['backend'], NumbaBackend)
    assert isinstance(clone(system).get_params()['backend'], NumbaBackend)


def test_prepare_passes_the_solver_and_the_backend_to_the_built_system() -> None:
    """The solver and the backend of a ready-made system reach the system it builds and the run."""
    system = MeadowSystem(solver=RK4Solver(dt=0.01), backend=NumbaBackend()).prepare()
    assert isinstance(system.solver_, RK4Solver)
    assert isinstance(system.backend_, NumbaBackend)
    assert isinstance(system.system_.backend, NumbaBackend)
    numba = system.simulate(until=1.0).get_values('meadow/prey.x')
    numpy = MeadowSystem(solver=RK4Solver(dt=0.01), backend=NumpyBackend()).prepare().simulate(until=1.0)
    assert numba == pytest.approx(numpy.get_values('meadow/prey.x'), rel=1e-9)


class TunedMeadowSystem(MeadowSystem):
    """System of one meadow whose built system chooses its own solver."""

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the meadow with the RK4 solver."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            systems=[LotkaVolterraSystem(label='meadow', r=self.r)],
            solver=RK4Solver(dt=0.01),
        )


def test_prepare_keeps_the_solver_of_the_built_system_for_auto() -> None:
    """A ready-made system left at `'auto'` keeps the solver its built system chooses, and a given one wins."""
    assert isinstance(TunedMeadowSystem().prepare().solver_, RK4Solver)
    assert TunedMeadowSystem(solver=RK4Solver(dt=0.02)).prepare().solver_.dt == 0.02


def test_set_params_changes_the_backend() -> None:
    """Setting the backend of a prepared ready-made system asks to prepare it again."""
    system = MeadowSystem().prepare().set_params(backend=NumbaBackend())
    with pytest.raises(NotPreparedError, match='parameters backend'):
        system.simulate(until=1.0)
    assert isinstance(system.prepare().backend_, NumbaBackend)


def test_prepare_exposes_what_the_built_system_learned() -> None:
    """A ready-made system exposes the states, equations, form, and solver of the system it builds."""
    system = MeadowSystem().prepare()
    assert system.states_ == ('meadow/prey.x', 'meadow/predators.x')
    assert system.form_ == 'ordinary differential equations'
    assert system.equations_ == system.system_.equations_
    assert system.solver_ is system.system_.solver_


class CountInteraction(Interaction):
    """Interaction that adds one to a count at every step."""

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role = 'counter') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next count as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target.n, context.target.n + 1)]


def test_get_tags_reads_the_domains_of_values_from_the_laws() -> None:
    """The tags of a system whose element is given by values name the domains of the states its laws change."""
    counter = Element('counter', values={'n': 0.0, 'step': 0.5})
    tags = System(elements=[counter], interactions=[CountInteraction('count')]).get_tags()
    assert tags.domains == {'real'}
    assert tags.laws == {'update'}


def test_get_tags_reads_the_laws_of_a_one_off_interaction_from_the_census() -> None:
    """The tags of a system name the kinds of laws a one-off interaction learns only when it is prepared."""
    flip = Interaction('flip', roles={'target': 'voter'}, laws=lambda s: [UpdateLaw(s.target.n, 1 - s.target.n)])
    tags = System(elements=[Element('voter', values={'n': 0})], interactions=[flip]).get_tags()
    assert flip.get_tags().laws is None
    assert (tags.laws, tags.time) == ({'update'}, {'discrete'})


def test_simulate_a_static_system_needs_no_end_time():
    """Test that a system that only minimises an objective simulates with no `until` to its one frame."""
    bowl = Interaction('bowl', roles={'target': 'p'}, laws=lambda s: [ObjectiveLaw((s.target.x - 2) ** 2, s.target.x)])
    trajectory = System(elements=[Element('p', values={'x': 0.0})], interactions=[bowl]).prepare().simulate()
    assert trajectory.times.tolist() == [0.0]
    assert trajectory.get_values('p.x')[-1] == pytest.approx(2.0, abs=1e-6)
