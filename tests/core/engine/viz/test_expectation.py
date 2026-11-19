"""Test the visual of operator elements and groups."""

from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.viz._expectation import ExpectationVisual
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.quantum import QubitElement


def test_local_values_give_probabilities_and_expectations():
    """Test the local values that visuals draw for single elements."""
    system = System(elements=[QubitElement('q', psi='down')])
    problem = OperatorCompiler().prepare(system._get_composite()).run().problem_
    values = ExpectationVisual._get_local_values(problem, 'q', problem.initial_state)
    assert values['z'] == -1.0
    assert values['x'] == 0.0
