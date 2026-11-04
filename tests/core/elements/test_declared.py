"""Test the `_declared` module of `skcomplex.core.elements`."""

import ast
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.elements import DeclaredElement, Element, FieldElement, GroupElement
from skcomplex.core.interactions import MapInteraction, RateInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.spaces import ContinuousTime, DiscreteTime, LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine._utils import declared_variables
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.experiments import BaseExperiment
from skcomplex.specs import DeclaredVariableSpec, ParameterSpec, RoleSpec, StateSpec

GENERATED_CLASSES_ALLOWED = {'engine/solvers/_mesa.py': 2}


class _AnnotatedDecayElement(Element):
    """Decaying element with annotated variables."""

    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 2.0,
        k: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.x = x
        self.y = y
        self.k = k

    def _get_laws(self, namespace):
        """Return the decay of both states."""
        return [RateLaw(namespace.x, -namespace.k * namespace.x), RateLaw(namespace.y, -namespace.k * namespace.y)]


class _DeclaredDecayElement(DeclaredElement):
    """Decaying element with the same variables declared from data."""

    def _get_laws(self, namespace):
        """Return the decay of both states."""
        return [RateLaw(namespace.x, -namespace.k * namespace.x), RateLaw(namespace.y, -namespace.k * namespace.y)]


def _variables():
    """Return the variables of the decaying element as data."""
    return {
        'x': DeclaredVariableSpec(1.0, StateSpec()),
        'y': DeclaredVariableSpec(2.0, StateSpec()),
        'k': DeclaredVariableSpec(0.5, ParameterSpec()),
    }


def _simulate(system, until=1.0):
    """Return the trajectory of a system."""
    return Simulation(system, dt=0.01).check().prepare().run(until=until).trajectory_


def _generated_classes(root):
    """Return the `__signature__` assignments and the calls of `type` with three arguments of a source tree."""
    found: dict[str, list[int]] = {}
    for path in sorted(root.rglob('*.py')):
        lines = []
        for node in ast.walk(ast.parse(path.read_text())):
            is_type = isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'type' and len(node.args) == 3
            is_signature = '__signature__' in {getattr(node, 'attr', None), getattr(node, 'value', None)}
            if is_type or is_signature:
                lines.append(node.lineno)
        if lines:
            found[path.relative_to(root).as_posix()] = sorted(lines)
    return found


def test_declared_element_gives_the_trajectory_of_annotated_states():
    """Test that states and a parameter given as data run as the same annotated ones do, within 1e-12."""
    annotated = _simulate(System(elements=[_AnnotatedDecayElement('cell')], time=ContinuousTime()))
    element = _DeclaredDecayElement('cell', variables=_variables())
    declared = _simulate(System(elements=[element], time=ContinuousTime()))
    for path in ('cell.x', 'cell.y'):
        np.testing.assert_allclose(declared.values(path), annotated.values(path), rtol=1e-12, atol=1e-12)


def test_declared_element_clone_is_independent():
    """Test that a clone keeps the states and specs, and that a change to the clone leaves the original unchanged."""
    original = DeclaredElement('cell', variables=_variables())
    copied = clone(original).set_params(k=0.25)
    assert list(declared_variables(copied)) == list(declared_variables(original))
    assert copied.variables['x'].spec == original.variables['x'].spec
    assert copied.get_value('k') == 0.25
    assert original.get_value('k') == 0.5


def test_declared_element_keeps_the_specs_of_annotated_arguments():
    """Test that an annotated class keeps its specs, and that an instance adds the ones it declares."""
    assert list(declared_variables(_AnnotatedDecayElement)) == ['x', 'y', 'k']
    assert list(declared_variables(_AnnotatedDecayElement('cell'))) == ['x', 'y', 'k']
    assert declared_variables(DeclaredElement) == {}
    assert list(declared_variables(DeclaredElement('cell', variables=_variables()))) == ['x', 'y', 'k']


def test_declared_element_reads_a_changed_value():
    """Test that the declarations of an element follow a change of a value after they were read."""
    element = DeclaredElement('cell', variables=_variables())
    assert declared_variables(element)['x'][1] == 1.0
    assert declared_variables(element.set_params(x=5.0))['x'][1] == 5.0
    assert declared_variables(element.set_params(variables={})) == {}


def test_declared_element_declares_the_companions_of_a_state():
    """Test that lags and derivatives of a declared state have the defaults of an annotated one."""
    specs = {
        'x': DeclaredVariableSpec(1.0, StateSpec(lags=2, initial=(0.5, 0.25))),
        'v': DeclaredVariableSpec(1.0, StateSpec(order=2, initial=(3.0,))),
    }
    variables = declared_variables(DeclaredElement('cell', variables=specs))
    assert list(variables) == ['x', 'x_lag1', 'x_lag2', 'v', 'v_d1']
    assert [variables[name][1] for name in ('x_lag1', 'x_lag2', 'v_d1')] == [0.5, 0.25, 3.0]


def test_declared_element_runs_a_map_on_its_lags():
    """Test that a map reads the lags of a declared state under discrete time."""
    specs = {'x': DeclaredVariableSpec(1.0, StateSpec(lags=1, initial=(0.0,)))}
    element = DeclaredElement('cell', variables=specs)
    fibonacci = MapInteraction('next', target='cell', state='x', mode='increment', function=lambda s: s.target.x_lag1)
    trajectory = Simulation(System(elements=[element], interactions=[fibonacci], time=DiscreteTime()), dt=1)
    assert trajectory.check().prepare().run(until=5).trajectory_.values('cell.x')[-1] == pytest.approx(8.0)


def test_declared_element_in_a_group_takes_values_per_member():
    """Test that a group of declared elements gives one value per member."""
    values = {'variables': _variables(), 'x': np.array([1.0, 2.0])}
    group = GroupElement('cells', element=_DeclaredDecayElement, size=2, values=values)
    last = _simulate(System(elements=[group], time=ContinuousTime())).values('cells.x')[-1]
    assert last == pytest.approx(np.array([1.0, 2.0]) * np.exp(-0.5), rel=1e-4)


def test_declared_element_on_a_field_runs_its_laws():
    """Test that a field of declared elements runs a state over the sites of a lattice."""
    space = LatticeSpace((3,))
    field = FieldElement('cells', element=_DeclaredDecayElement, space=space, values={'variables': _variables()})
    last = _simulate(System(elements=[field], space=space, time=ContinuousTime())).values('cells.x')[-1]
    assert np.ravel(last) == pytest.approx(np.full(3, np.exp(-0.5)), rel=1e-4)


def test_declared_element_replicates_in_an_experiment():
    """Test that replicating a system keeps the declared variables and takes one value per member."""
    element = _DeclaredDecayElement('cell', variables=_variables())
    system = System(elements=[element], time=ContinuousTime())
    ensemble = BaseExperiment._replicate(system, 2, {'cell.x': np.array([1.0, 3.0])})
    last = _simulate(ensemble).values('cell.x')[-1]
    assert last == pytest.approx(np.array([1.0, 3.0]) * np.exp(-0.5), rel=1e-4)


def test_declared_element_takes_a_rate_of_an_interaction():
    """Test that an interaction binds a declared state by its name."""
    element = DeclaredElement('cell', variables={'x': DeclaredVariableSpec(1.0, StateSpec())})
    rate = RateInteraction('growth', target='cell', state='x', rate=lambda s: s.target.x)
    trajectory = _simulate(System(elements=[element], interactions=[rate], time=ContinuousTime()))
    assert trajectory.values('cell.x')[-1] == pytest.approx(np.e, rel=1e-4)


@pytest.mark.parametrize('name', ['not an identifier', 'label', 'variables'])
def test_declared_element_rejects_a_name_a_law_cannot_read(name):
    """Test that a declared name must be an identifier that no argument of the constructor takes."""
    with pytest.raises(ValidationError, match='declares'):
        DeclaredElement('cell', variables={name: DeclaredVariableSpec(1.0, StateSpec())}).check()


def test_declared_element_rejects_a_role():
    """Test that a role is not a variable an element declares from data."""
    with pytest.raises(ValidationError, match='StateSpec or a ParameterSpec'):
        DeclaredElement('cell', variables={'x': DeclaredVariableSpec(1.0, RoleSpec())}).check()


def test_declared_element_rejects_an_unknown_name_to_set():
    """Test that setting a name that is neither an argument nor a declared variable raises the error of `set_params`."""
    with pytest.raises(ValueError, match='nothing'):
        DeclaredElement('cell', variables=_variables()).set_params(nothing=1.0)


def test_library_generates_no_classes_of_parts():
    """Test that no module sets `__signature__` and that only the agents of Mesa are built with `type`."""
    found = _generated_classes(Path(str(files('skcomplex'))))
    assert {path: len(lines) for path, lines in found.items()} == GENERATED_CLASSES_ALLOWED


def test_generated_classes_finds_a_planted_signature(tmp_path):
    """Test that the source check finds a planted `__signature__` and a planted class built with `type`."""
    (tmp_path / 'planted.py').write_text(
        "vars(f)['__signature__'] = 1\nf.__signature__ = 2\nC = type('C', (object,), {})\nd = type(object())\n",
    )
    assert _generated_classes(tmp_path) == {'planted.py': [1, 2, 3]}
