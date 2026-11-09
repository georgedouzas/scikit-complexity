"""Test the `_line_automata` module of `skcomplex.systems.computer_science.automata`."""

import itertools

import numpy as np
import pytest

from skcomplex.core.errors import ParameterError
from skcomplex.systems.computer_science.automata import InvolutiveAutomatonSystem, NilpotentAutomatonSystem

from ..conftest import _run


@pytest.mark.parametrize('n', [2, 3, 4])
def test_nilpotent_automaton_reaches_zero_in_exactly_n_steps(n):
    """Test that `n` steps send every ring of ten cells to 0, and that `n - 1` steps leave the run of `n + 1` ones."""
    configurations = np.asarray(list(itertools.product((0, 1), repeat=10)), dtype=float)[::7]
    for configuration in configurations:
        values = _run(NilpotentAutomatonSystem(n, initial=configuration), n).get_values('cells.alive')
        assert not values[-1].any()
    witness = np.zeros(12)
    witness[2 : n + 3] = 1
    values = _run(NilpotentAutomatonSystem(n, initial=witness), n).get_values('cells.alive')
    assert values[-2].any()
    assert not values[-1].any()


def test_involutive_automaton_is_its_own_inverse():
    """Test that two steps return every configuration of a ring to itself, and that alpha and beta swap."""
    for configuration in np.asarray(list(itertools.product((0, 1), repeat=11)), dtype=float)[::5]:
        values = _run(InvolutiveAutomatonSystem(3, initial=configuration), 2).get_values('cells.alive')
        np.testing.assert_array_equal(values[2], values[0])
    alpha = np.zeros(10)
    alpha[3:6] = 1
    image = _run(InvolutiveAutomatonSystem(3, initial=alpha), 1).get_values('cells.alive')[-1]
    np.testing.assert_array_equal(image, np.r_[np.zeros(4), np.ones(2), np.zeros(4)])


def test_line_automata_reject_short_memories():
    """Test that classes and run lengths below their minimum raise a parameter error."""
    with pytest.raises(ParameterError, match='at least 2'):
        NilpotentAutomatonSystem(1).prepare()
    with pytest.raises(ParameterError, match='at least 3'):
        InvolutiveAutomatonSystem(2).prepare()
