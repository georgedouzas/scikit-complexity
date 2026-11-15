"""Test the `_system` module of `skcomplex.core.systems`."""

import re
from types import SimpleNamespace

import networkx as nx
import pytest
from sklearn.base import clone

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import mean, neighbors
from skcomplex.core.components.spaces import BaseSpace, NetworkSpace
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System
from skcomplex.systems.biology.ecology import LotkaVolterraSystem


def meadow_system(solver: RK4Solver | str = 'auto') -> System:
    """Return a system of one nested Lotka-Volterra meadow."""
    return System('ecology', systems=[LotkaVolterraSystem(label='meadow')], solver=solver)


def test_get_params_lists_nested_names() -> None:
    """The parameters of a nested component appear under its label."""
    params = meadow_system().get_params()
    assert params['meadow__r'] == pytest.approx(1.1)
    assert params['solver'] == 'auto'


def test_set_params_tunes_component() -> None:
    """A nested name sets the parameter of the component with that label."""
    system = meadow_system().set_params(meadow__r=1.5)
    assert system.systems[0].r == pytest.approx(1.5)
    assert system.get_params()['meadow__r'] == pytest.approx(1.5)


def test_set_params_unknown_component_raises() -> None:
    """A nested name that matches no component raises an error that lists the components."""
    with pytest.raises(ValidationError, match=r"no component labelled 'forest'.*'meadow'"):
        meadow_system().set_params(forest__r=1.0)


def test_set_params_unknown_parameter_raises() -> None:
    """A nested name whose parameter the component lacks raises an error that names both."""
    with pytest.raises(ValidationError, match=r"'meadow'.*no parameter 'mass'"):
        meadow_system().set_params(meadow__mass=2.0)


def test_set_params_unknown_own_parameter_raises() -> None:
    """A plain name that is not a parameter of the system raises an error."""
    with pytest.raises(ValidationError, match="no parameter 'time'"):
        meadow_system().set_params(time=1.0)


def test_clone_keeps_components() -> None:
    """Cloning a system keeps its label, components, and settings."""
    copy = clone(meadow_system(solver=RK4Solver()))
    assert copy.label == 'ecology'
    assert isinstance(copy.solver, RK4Solver)
    assert copy.get_params()['meadow__prey'] == pytest.approx(40.0)


def test_prepare_compiles_once_for_many_runs() -> None:
    """Two runs of one prepared system keep what prepare learned."""
    system = meadow_system().prepare()
    equations = system.equations_
    system.simulate(until=1.0)
    system.simulate(until=2.0)
    assert system.equations_ is equations


def test_prepare_sets_learned_attributes() -> None:
    """A prepared system names its states and parameters and shows its equations and form."""
    system = meadow_system().prepare()
    assert system.states_ == ('meadow/prey.x', 'meadow/predators.x')
    assert 'meadow/growth.r' in system.parameters_
    predators = 'meadow/predation.b*meadow/predators.x*meadow/prey.x - meadow/death.d*meadow/predators.x'
    assert system.equations_['meadow/predators.x'] == predators
    assert system.form_ == 'ordinary differential equations'
    assert type(system.backend_).__name__ == 'NumpyBackend'


def test_prepare_names_nested_states_by_path() -> None:
    """The states of a nested system are named by their path from the system that holds it."""
    parent = System('world', systems=[meadow_system()]).prepare()
    assert parent.states_ == ('ecology/meadow/prey.x', 'ecology/meadow/predators.x')


def test_prepare_uses_named_solver() -> None:
    """A solver given to the system runs the problem."""
    system = meadow_system(solver=RK4Solver()).prepare()
    assert isinstance(system.solver_, RK4Solver)
    assert system.simulate(until=1.0, dt=0.01).times[-1] == pytest.approx(1.0)


def test_prepare_rejects_bad_solver() -> None:
    """A solver that is neither 'auto' nor a solver raises an error."""
    with pytest.raises(ValidationError, match="solver of the system 'ecology' must be 'auto'"):
        meadow_system(solver='fast').prepare()


def test_nested_system_runs_in_parent() -> None:
    """A system nested in another one runs inside it."""
    parent = System('world', systems=[meadow_system()]).prepare()
    assert 'ecology/meadow/prey.x' in parent.simulate(until=1.0).names


class _Consensus(Interaction):
    """Interaction that moves every member toward its neighbours, or toward the mean of its group."""

    def __init__(self, label: str, *, target: Role = 'nodes', over: str = 'neighbours') -> None:
        super().__init__(label)
        self.target = target
        self.over = over

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull toward the neighbours or toward the mean."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            pull = neighbors(context.target.other.x - x) if self.over == 'neighbours' else mean(x) - x
            self.laws_: list[BaseComponent] = [RateLaw(x.diff(context.t), pull)]


def test_prepare_writes_reductions_readably():
    """Test that sums over neighbours and means over a group appear in the equations by the states they read."""
    nodes = Element('nodes', values={'x': [1.0, 0.0, 0.0]}, space=NetworkSpace(nx.path_graph(3)))
    crowd = Element('crowd', values={'x': [1.0, 2.0, 6.0]}, size=3)
    system = System(
        elements=[nodes, crowd],
        interactions=[_Consensus('consensus'), _Consensus('pull', target='crowd', over='group')],
    ).prepare()
    assert system.equations_['nodes.x[i]'] == "neighbors(-nodes.x[i] + nodes.x[j], reduce='sum')"
    assert system.equations_['crowd.x[i]'] == 'mean(crowd.x) - crowd.x[i]'
    assert not any(re.search(r'(?<![\w.])_\w', equation) for equation in system.equations_.values())
