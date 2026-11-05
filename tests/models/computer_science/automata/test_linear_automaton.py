"""Test the `_linear_automaton` module of `skcomplex.models.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.spaces import BoxNeighborhood, LatticeNeighborhood, LatticeSpace
from skcomplex.errors import ValidationError
from skcomplex.models.computer_science.automata import LinearAutomaton


def test_rule_90_matrix_and_step_agree():
    """Test that rule 90 on a ring of 6 cells steps as its matrix over GF(2) maps."""
    rule90 = LinearAutomaton(LatticeSpace(6), LatticeNeighborhood(((-1,), (1,))), (1, 1), 'GF(2)')
    configuration = np.array([1, 0, 1, 1, 0, 0])
    entries = np.asarray([[int(entry) for entry in row] for row in rule90.matrix().rows()])
    np.testing.assert_array_equal(rule90.step(configuration), entries @ configuration % 2)


def test_laurent_polynomial_of_the_laplacian():
    """Test that the Laplacian over QQ has the symbol 1/x - 2 + x."""
    laplacian = LinearAutomaton(LatticeSpace(5), BoxNeighborhood(), (1, -2, 1), 'QQ')
    polynomial = laplacian.laurent_polynomial()
    assert sorted(polynomial.exponents()) == [-1, 0, 1]
    assert polynomial(1) == 0


def test_unknown_field_and_ball_raise():
    """Test that an unknown field and a matrix on a lattice with fixed edges are rejected."""
    with pytest.raises(ValidationError, match='Unknown field'):
        LinearAutomaton(LatticeSpace(5), BoxNeighborhood(), (1, 1, 1), 'GF(x)').step(np.zeros(5))
    fixed = LinearAutomaton(LatticeSpace(5, boundary='fixed'), BoxNeighborhood(), (1, 1, 1))
    with pytest.raises(ValidationError, match='not on a ball'):
        fixed.matrix()


def test_reference_values_of_linear_rules():
    """Test that linear automata keep their recorded steps, matrices, and Laurent polynomials."""
    rule90 = LinearAutomaton(LatticeSpace(6), LatticeNeighborhood(((-1,), (1,))), (1, 1), 'GF(2)')
    assert rule90.step([0, 0, 1, 0, 0, 0]).tolist() == [0, 1, 0, 1, 0, 0]
    assert rule90.step([1, 0, 1, 1, 0, 0]).tolist() == [0, 0, 1, 1, 1, 1]
    assert [[int(entry) for entry in row] for row in rule90.matrix().rows()] == [
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ]
    assert rule90.matrix().rank() == 4
    assert str(rule90.laurent_polynomial()) == 'x0^-1 + x0'
    laplacian = LinearAutomaton(LatticeSpace(5), BoxNeighborhood(), (1, -2, 1), 'QQ')
    assert laplacian.step([1, 0, 0, 1, 0]).tolist() == [-2, 1, 1, -2, 2]
    assert str(laplacian.laurent_polynomial()) == 'x0^-1 - 2 + x0'
    assert laplacian.matrix().rank() == 4
    fractions = LinearAutomaton(LatticeSpace(3), LatticeNeighborhood(((0,), (1,))), (0.5, 0.25), 'QQ')
    assert fractions.step([1, 2, 4]).tolist() == [1.0, 2.0, 2.25]
    modular = LinearAutomaton(LatticeSpace(5), BoxNeighborhood(), (1, -2, 1), 'GF(3)')
    assert modular.step([1, 0, 0, 2, 0]).tolist() == [1, 1, 2, 2, 0]
    assert str(modular.laurent_polynomial()) == 'x0^-1 + 1 + x0'
    assert [int(entry) for entry in modular.matrix().rows()[0]] == [1, 1, 0, 0, 1]
    plane = LinearAutomaton(LatticeSpace((3, 3)), LatticeNeighborhood(((1, 0), (0, -1))), (1, 1), 'GF(2)')
    assert str(plane.laurent_polynomial()) == 'x0 + x1^-1'
