"""Test the `_declared` module of `skcomplex.core.components.elements`."""

from typing import Annotated

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.components import RoleSpec
from skcomplex.core.components._declared_variable_spec import _DeclaredVariableSpec
from skcomplex.core.components._parameter_spec import _ParameterSpec
from skcomplex.core.components._state_spec import _StateSpec
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.spaces._continuous_time import _ContinuousTime
from skcomplex.core.engine._utils import declared_variables
from skcomplex.core.engine.problems._composite_system import _CompositeSystem
from skcomplex.core.engine.problems._declared_element import _DeclaredElement
from skcomplex.core.engine.problems._field_element import _FieldElement
from skcomplex.core.engine.problems._group_element import _GroupElement
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments._replicate import _replicate


class _AnnotatedDecayElement(Element):
    """Decaying element with annotated variables."""

    def __init__(
        self,
        label,
        *,
        x: Annotated[float, _StateSpec()] = 1.0,
        y: Annotated[float, _StateSpec()] = 2.0,
        k: Annotated[float, _ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.x = x
        self.y = y
        self.k = k

    def _get_laws(self, namespace):
        """Return the decay of both states."""
        return [RateLaw(namespace.x, -namespace.k * namespace.x), RateLaw(namespace.y, -namespace.k * namespace.y)]


class _DeclaredDecayElement(_DeclaredElement):
    """Decaying element with the same variables declared from data."""

    def _get_laws(self, namespace):
        """Return the decay of both states."""
        return [RateLaw(namespace.x, -namespace.k * namespace.x), RateLaw(namespace.y, -namespace.k * namespace.y)]


def _variables():
    """Return the variables of the decaying element as data."""
    return {
        'x': _DeclaredVariableSpec(1.0, _StateSpec()),
        'y': _DeclaredVariableSpec(2.0, _StateSpec()),
        'k': _DeclaredVariableSpec(0.5, _ParameterSpec()),
    }


def _simulate(system, until=1.0):
    """Return the trajectory of a system."""
    return simulate(system, until=until, dt=0.01)


def test_declared_element_gives_the_trajectory_of_annotated_states():
    """Test that states and a parameter given as data run as the same annotated ones do, within 1e-12."""
    annotated = _simulate(_CompositeSystem(elements=[_AnnotatedDecayElement('cell')], time=_ContinuousTime()))
    element = _DeclaredDecayElement('cell', variables=_variables())
    declared = _simulate(_CompositeSystem(elements=[element], time=_ContinuousTime()))
    for path in ('cell.x', 'cell.y'):
        np.testing.assert_allclose(declared.get_values(path), annotated.get_values(path), rtol=1e-12, atol=1e-12)


def test_declared_element_clone_is_independent():
    """Test that a clone keeps the states and specs, and that a change to the clone leaves the original unchanged."""
    original = _DeclaredElement('cell', variables=_variables())
    copied = clone(original).set_params(k=0.25)
    assert list(declared_variables(copied)) == list(declared_variables(original))
    assert copied.variables['x'].spec == original.variables['x'].spec
    assert copied.get_value('k') == 0.25
    assert original.get_value('k') == 0.5


def test_declared_element_keeps_the_specs_of_annotated_arguments():
    """Test that an annotated class keeps its specs, and that an instance adds the ones it declares."""
    assert list(declared_variables(_AnnotatedDecayElement)) == ['x', 'y', 'k']
    assert list(declared_variables(_AnnotatedDecayElement('cell'))) == ['x', 'y', 'k']
    assert declared_variables(_DeclaredElement) == {}
    assert list(declared_variables(_DeclaredElement('cell', variables=_variables()))) == ['x', 'y', 'k']


def test_declared_element_reads_a_changed_value():
    """Test that the declarations of an element follow a change of a value after they were read."""
    element = _DeclaredElement('cell', variables=_variables())
    assert declared_variables(element)['x'][1] == 1.0
    assert declared_variables(element.set_params(x=5.0))['x'][1] == 5.0
    assert declared_variables(element.set_params(variables={})) == {}


def test_declared_element_declares_the_companions_of_a_state():
    """Test that lags and derivatives of a declared state have the defaults of an annotated one."""
    specs = {
        'x': _DeclaredVariableSpec(1.0, _StateSpec(lags=2, initial=(0.5, 0.25))),
        'v': _DeclaredVariableSpec(1.0, _StateSpec(order=2, initial=(3.0,))),
    }
    variables = declared_variables(_DeclaredElement('cell', variables=specs))
    assert list(variables) == ['x', 'x_lag1', 'x_lag2', 'v', 'v_d1']
    assert [variables[name][1] for name in ('x_lag1', 'x_lag2', 'v_d1')] == [0.5, 0.25, 3.0]


def test_declared_element_sets_the_initial_values_of_its_companions():
    """Test that setting a companion state changes the initial value the spec of its state holds."""
    specs = {
        'x': _DeclaredVariableSpec(1.0, _StateSpec(lags=2)),
        'v': _DeclaredVariableSpec(1.0, _StateSpec(order=3, initial=(3.0,))),
    }
    element = _DeclaredElement('cell', variables=specs)
    assert [element.get_value(name) for name in ('x_lag1', 'x_lag2', 'v_d1', 'v_d2')] == [1.0, 1.0, 3.0, 0.0]
    element.set_params(x_lag2=0.5, v=2.0, v_d2=4.0)
    assert element.variables['x'].spec.initial == (1.0, 0.5)
    assert element.variables['v'] == _DeclaredVariableSpec(2.0, _StateSpec(order=3, initial=(3.0, 4.0)))
    variables = declared_variables(element)
    assert [variables[name][1] for name in ('x_lag1', 'x_lag2', 'v_d1', 'v_d2')] == [1.0, 0.5, 3.0, 4.0]


def test_declared_element_replicates_the_derivatives_of_a_state():
    """Test that replicating a system takes one initial derivative of a declared state per member."""
    element = _DeclaredElement('mass', variables={'x': _DeclaredVariableSpec(0.0, _StateSpec(order=2))})
    system = _CompositeSystem(elements=[element], time=_ContinuousTime())
    ensemble = _replicate(system, 2, {'mass.x_d1': np.array([1.0, -2.0])})
    assert _simulate(ensemble).get_values('mass.x')[-1] == pytest.approx(np.array([1.0, -2.0]), rel=1e-6)


def test_declared_element_in_a_group_takes_values_per_member():
    """Test that a group of declared elements gives one value per member."""
    values = {'variables': _variables(), 'x': np.array([1.0, 2.0])}
    group = _GroupElement('cells', element=_DeclaredDecayElement, size=2, values=values)
    last = _simulate(_CompositeSystem(elements=[group], time=_ContinuousTime())).get_values('cells.x')[-1]
    assert last == pytest.approx(np.array([1.0, 2.0]) * np.exp(-0.5), rel=1e-4)


def test_declared_element_on_a_field_runs_its_laws():
    """Test that a field of declared elements runs a state over the sites of a lattice."""
    space = LatticeSpace((3,))
    field = _FieldElement('cells', element=_DeclaredDecayElement, space=space, values={'variables': _variables()})
    last = _simulate(_CompositeSystem(elements=[field], space=space, time=_ContinuousTime())).get_values('cells.x')[-1]
    assert np.ravel(last) == pytest.approx(np.full(3, np.exp(-0.5)), rel=1e-4)


def test_declared_element_replicates_in_an_experiment():
    """Test that replicating a system keeps the declared variables and takes one value per member."""
    element = _DeclaredDecayElement('cell', variables=_variables())
    system = _CompositeSystem(elements=[element], time=_ContinuousTime())
    ensemble = _replicate(system, 2, {'cell.x': np.array([1.0, 3.0])})
    last = _simulate(ensemble).get_values('cell.x')[-1]
    assert last == pytest.approx(np.array([1.0, 3.0]) * np.exp(-0.5), rel=1e-4)


def test_declared_element_takes_a_rate_of_an_interaction():
    """Test that an interaction binds a declared state by its name."""
    element = _DeclaredElement('cell', variables={'x': _DeclaredVariableSpec(1.0, _StateSpec())})
    rate = Interaction('growth', roles={'target': 'cell'}, laws=lambda s: [RateLaw(s.target.x, s.target.x)])
    trajectory = _simulate(_CompositeSystem(elements=[element], interactions=[rate], time=_ContinuousTime()))
    assert trajectory.get_values('cell.x')[-1] == pytest.approx(np.e, rel=1e-4)


@pytest.mark.parametrize('name', ['not an identifier', 'label', 'variables'])
def test_declared_element_rejects_a_name_a_law_cannot_read(name):
    """Test that a declared name must be an identifier that no argument of the constructor takes."""
    with pytest.raises(ValidationError, match='declares'):
        _DeclaredElement('cell', variables={name: _DeclaredVariableSpec(1.0, _StateSpec())}).prepare()


def test_declared_element_rejects_a_role():
    """Test that a role is not a variable an element declares from data."""
    with pytest.raises(ValidationError, match='StateSpec or a ParameterSpec'):
        _DeclaredElement('cell', variables={'x': _DeclaredVariableSpec(1.0, RoleSpec())}).prepare()


def test_declared_element_rejects_an_unknown_name_to_set():
    """Test that setting a name that is neither an argument nor a declared variable raises the error of `set_params`."""
    with pytest.raises(ValueError, match='nothing'):
        _DeclaredElement('cell', variables=_variables()).set_params(nothing=1.0)


def test_get_states_returns_the_states_declared_from_data():
    """Test that a declared element returns the states an instance declares, and no parameter."""
    variables = {'x': _DeclaredVariableSpec(1.0, _StateSpec()), 'r': _DeclaredVariableSpec(0.5, _ParameterSpec())}
    assert list(_DeclaredElement('cell', variables=variables).get_states()) == ['x']
