"""Test the visual of quantum elements and groups."""

from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.viz._quantum import local_values
from skcomplex.models.physics.common.quantum import QubitElement


def test_local_values_give_probabilities_and_expectations():
    """Test the local values that visuals draw for single elements."""
    problem = compile_system(System(elements=[QubitElement('q', psi='down')]))
    values = local_values(problem, 'q', problem.initial_state)
    assert values['z'] == -1.0
    assert values['x'] == 0.0
