"""Test the visual of operator elements and groups."""

from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.engine.viz._expectation import ExpectationVisual
from skcomplex.systems.physics.common.quantum import QubitElement


def test_local_values_give_probabilities_and_expectations():
    """Test the local values that visuals draw for single elements."""
    problem = OperatorCompiler().prepare(System(elements=[QubitElement('q', psi='down')])).run().problem_
    values = ExpectationVisual._local_values(problem, 'q', problem.initial_state)
    assert values['z'] == -1.0
    assert values['x'] == 0.0
