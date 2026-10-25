"""Test the visual of quantum elements and groups."""

from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.formalisms.quantum._visuals import local_values
from skcomplex.models.physics.realizations.quantum import QubitComponent


def test_local_values_give_probabilities_and_expectations():
    """Test the local values that visuals draw for single elements."""
    problem = compile_system(System(components=[QubitComponent('q', initial='down')]))
    values = local_values(problem, 'q', problem.initial_state)
    assert values['z'] == -1.0
    assert values['x'] == 0.0
