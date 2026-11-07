"""Test the `_free_group_automata` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.errors import ParameterError
from skcomplex.systems.computer_science.automata import (
    LifeInATreeSystem,
    MullerAutomatonSystem,
    SurjectiveLinearAutomatonSystem,
)

from ..conftest import _run


def _pairs(system):
    """Return the first and second coordinates after one step."""
    trajectory = _run(system, 1)
    return trajectory.values('cells.first')[-1], trajectory.values('cells.second')[-1]


def test_muller_automaton_has_its_image_in_the_first_factor():
    """Test that the second coordinate of every image is 0, so the automaton is not surjective."""
    _, second = _pairs(MullerAutomatonSystem(radius=3, modulus=3))
    np.testing.assert_array_equal(second, 0)


def test_muller_automaton_reads_the_projections_at_the_generators():
    """Test that a cell becomes `h(ga) + k(gb) + h(ga^-1) + k(gb^-1)`, read through the generator table."""
    system = MullerAutomatonSystem(radius=3, modulus=5).prepare().system_
    space = system.elements[0].space.prepare()
    trajectory = _run(system, 1)
    first, second = trajectory.values('cells.first')[0], trajectory.values('cells.second')[0]
    padded_first, padded_second = np.append(first, 0), np.append(second, 0)
    table = np.where(space.generator_table_ >= 0, space.generator_table_, space.size_)
    expected = padded_first[table[:, 0]] + padded_first[table[:, 1]] + padded_second[table[:, 2]]
    expected = (expected + padded_second[table[:, 3]]) % 5
    np.testing.assert_array_equal(trajectory.values('cells.first')[-1], expected)


def test_muller_automaton_sends_no_finite_nonzero_configuration_to_zero():
    """Test that nonzero configurations supported well inside the ball have nonzero images, the pre-injectivity."""
    system = MullerAutomatonSystem(radius=4).prepare().system_
    space = system.elements[0].space.prepare()
    interior = np.flatnonzero(space.lengths_ <= 2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        pairs = np.zeros((space.size_, 2))
        pairs[interior] = rng.integers(0, 2, (interior.size, 2))
        if not pairs.any():
            continue
        first, _ = _pairs(MullerAutomatonSystem(radius=4, initial=pairs))
        assert first.any()


def test_surjective_linear_automaton_kills_a_vector_at_the_identity():
    """Test that `(0, 1)` at the identity and 0 elsewhere maps to 0, so the automaton is not pre-injective."""
    pairs = np.zeros((161, 2))
    pairs[0, 1] = 1
    first, second = _pairs(SurjectiveLinearAutomatonSystem(radius=4, initial=pairs))
    assert not first.any()
    assert not second.any()


def test_surjective_linear_automaton_sums_the_first_coordinates():
    """Test that the image is `(h(ga) + h(ga^-1), h(gb) + h(gb^-1))` modulo the field size."""
    system = SurjectiveLinearAutomatonSystem(radius=3, modulus=3).prepare().system_
    space = system.elements[0].space.prepare()
    trajectory = _run(system, 1)
    first = np.append(trajectory.values('cells.first')[0], 0)
    table = np.where(space.generator_table_ >= 0, space.generator_table_, space.size_)
    np.testing.assert_array_equal(trajectory.values('cells.first')[-1], (first[table[:, 0]] + first[table[:, 1]]) % 3)
    np.testing.assert_array_equal(trajectory.values('cells.second')[-1], (first[table[:, 2]] + first[table[:, 3]]) % 3)


def test_pair_automata_reject_a_trivial_group():
    """Test that a modulus below 2 raises a parameter error."""
    with pytest.raises(ParameterError, match='at least 2'):
        MullerAutomatonSystem(modulus=1).prepare()


def test_life_in_a_tree_follows_the_rule_of_the_exercise():
    """Test that a cell lives when the sum over itself and its eight neighbours is 3, or 4 with the cell alive."""
    system = LifeInATreeSystem(radius=3, density=0.4).prepare().system_
    space = system.elements[0].space.prepare()
    trajectory = _run(system, 1)
    cells = np.append(trajectory.values('cells.alive')[0], 0)
    table = np.where(space.generator_table_ >= 0, space.generator_table_, space.size_)
    total = cells[:-1] + cells[table].sum(axis=1)
    expected = (total == 3) | ((total == 4) & (cells[:-1] == 1))
    np.testing.assert_array_equal(trajectory.values('cells.alive')[-1], expected)
    assert space.size_ == 1 + 8 + 8 * 7 + 8 * 49
