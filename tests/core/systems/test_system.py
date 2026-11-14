"""Test the `_system` module of `skcomplex.core.systems`."""

import pytest
from sklearn.base import clone

from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
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
    """Two runs of one prepared system share the problem compiled by prepare."""
    system = meadow_system().prepare()
    problem = system.problem_
    system.simulate(until=1.0)
    system.simulate(until=2.0)
    assert system.problem_ is problem


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
