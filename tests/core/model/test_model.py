"""Test the description layer: components, laws, symbols, spaces, and systems."""

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import LabelError, ValidationError, state
from skcomplex.core.compile import compile_system
from skcomplex.core.model import AgentType, ContinuousSpace, Element, Jump, Rate, Symbols, System
from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.ecology import Population, Predation


def test_element_stores_arguments():
    """Test that an element stores its label and initial state unmodified."""
    element = Population('rabbits', n=40)
    assert element.label == 'rabbits'
    assert element.n == 40


def test_interaction_requires_roles():
    """Test that an interaction without its roles cannot be created."""
    with pytest.raises(TypeError, match='predator'):
        Predation('hunt', prey='rabbits')


def test_component_supports_estimator_contract():
    """Test that components follow the scikit-learn parameter contract."""
    interaction = Predation('hunt', predator='foxes', prey='rabbits', a=0.4)
    assert interaction.get_params()['a'] == 0.4
    cloned = clone(interaction)
    assert cloned is not interaction
    assert cloned.a == 0.4


def test_symbols_unknown_name_lists_available_names():
    """Test that a missing symbol names the available ones."""
    with pytest.raises(AttributeError, match='Available names are: a'):
        _ = Symbols(a=1).b


def test_system_repr_counts_children():
    """Test that a system shows its label and the number of each child kind."""
    system = System('meadow', components=[Population('rabbits')], space=ContinuousSpace(n_dim=2))
    assert repr(system) == "System('meadow', components=1, interactions=0)"


def test_element_state_named_visual_raises_error():
    """Test that an element type with a state named visual names the state and lists the reserved names."""
    with pytest.raises(LabelError, match=r"'visual'.*label, visual are reserved"):
        Element.define('Painted', states={'visual': 0.0})


@pytest.mark.parametrize('name', ['location', 'alive'])
def test_agent_state_named_like_view_raises_error(name):
    """Test that an agent type with a state named like an attribute of the agent view raises a label error."""
    with pytest.raises(LabelError, match=rf"'{name}'.*agents_here.*reserved"):
        type('Settler', (AgentType,), {'__annotations__': {name: float}, name: state(0.0)})


def test_agent_state_named_position_raises_error():
    """Test that an agent type with a state that Mesa reads, such as position, raises a named error at once."""
    with pytest.raises(LabelError, match=r"Walker declares the variable 'position'"):
        type('Walker', (AgentType,), {'__annotations__': {'position': float}, 'position': state(0.0)})


def _numpy_vector_rate(s):
    """Return a rate of a vector state written with a NumPy scalar factor."""
    return [Rate(s.v, np.float64(2.0) * s.v * s.v)]


def test_numpy_scalar_times_vector_is_rejected():
    """Test that a law value that NumPy built as an array raises a named error that names the state."""
    Moving = Element.define('Moving', states={'v': state(0.0, vector=True)}, laws=_numpy_vector_rate)
    system = System(components=[Moving('m', v=[1.0, 2.0])], space=ContinuousSpace(n_dim=2))
    with pytest.raises(ValidationError, match=r'rate law of \(v_m.*float'):
        compile_system(system)


def _numpy_decay(s):
    """Return a decay and a constant inflow written with NumPy scalars."""
    return [Rate(s.x, np.float64(-0.5) * s.x), Rate(s.x, np.float64(1.0))]


def _float_decay(s):
    """Return a decay and a constant inflow written with Python floats."""
    return [Rate(s.x, -0.5 * s.x), Rate(s.x, 1.0)]


def test_numpy_scalar_values_match_floats():
    """Test that NumPy scalar law values run like the same laws written with Python floats."""
    trajectories = [
        Simulation(System(components=[Element.define('Decay', states={'x': 2.0}, laws=laws)('d')]), dt=0.5)
        .run(until=2)
        .trajectory_['d.x']
        for laws in (_numpy_decay, _float_decay)
    ]
    assert np.array_equal(*trajectories)


def _parameter_jump(s):
    """Return a jump whose increment is a parameter."""
    return [Jump(s.n, {s.n: s.k})]


def test_jump_increment_symbol_is_rejected():
    """Test that a jump whose increment is a parameter raises a named error naming the state and the increment."""
    Counter = Element.define('Counter', states={'n': 0.0}, parameters={'k': 2.0}, laws=_parameter_jump)
    with pytest.raises(ValidationError, match=r'jump of n_c has the increment k_c.*number'):
        compile_system(System(components=[Counter('c')]))
