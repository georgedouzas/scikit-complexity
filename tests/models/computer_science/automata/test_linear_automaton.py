"""Test the `_linear_automaton` module of `skcomplex.models.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.spaces import BoxNeighborhood, LatticeNeighborhood, LatticeSpace
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
