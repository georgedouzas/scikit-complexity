"""Test group spaces: their sites, neighbour tables, and memory sets, and local rules run on them."""

import sys

import numpy as np
import pytest

from skcomplex.core.base import NumericDomain, SpaceKind, SpaceTooLargeError, ValidationError, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import Table, Totalistic
from skcomplex.core.space import LatticeSpace, Memory, Neighborhood, neighbor_pairs
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.groups import (
    CayleySpace,
    automaton_group,
    finitely_presented_group,
    matrix_group,
    permutation_group,
)
from skcomplex.formalisms.lattices import LocalRuleLaw

HEISENBERG = ([[1, 1, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
GRIGORCHUK = {
    'a': ((None, None), (1, 0)),
    'b': (('a', 'c'), None),
    'c': (('a', 'd'), None),
    'd': ((None, 'b'), None),
}
MAJORITY = Totalistic({3: 1, 4: 1, 5: 1})
FREE_MEMORY = Memory(('1', 'a', 'a^-1', 'b', 'b^-1'))


class CellComponent(ElementComponent):
    """Cell that takes the majority value of itself and its generators and their inverses."""

    alive: float = state(default=0, domain=NumericDomain.BOOLEAN)

    def laws(self, s):
        """Return the majority rule."""
        return [LocalRuleLaw(s.alive, FREE_MEMORY, MAJORITY)]


class CopyComponent(ElementComponent):
    """Cell that copies the value at the generator `a`."""

    alive: float = state(default=0, domain=NumericDomain.BOOLEAN)

    def laws(self, s):
        """Return the copy of the cell at `g a`."""
        return [LocalRuleLaw(s.alive, Memory(('a',)), Table({(1,): 1}))]


def _run(system, until, backend='numpy'):
    """Return the trajectory of a system."""
    return Simulation(system, dt=1, random_state=0, backend=backend).run(until=until).trajectory_


def test_memory_generators_lists_the_identity_and_inverses():
    """Test that the memory set of the generators holds the identity first and every generator with its inverse."""
    group = finitely_presented_group(['a', 'b'])
    assert Memory.generators(group).elements == ('1', 'a', 'a^-1', 'b', 'b^-1')
    assert Memory.generators(group, center=False, inverses=False).elements == ('a', 'b')


def test_group_space_is_a_lattice_kind_space():
    """Test that a group space declares the lattice kind and lacks a regular geometry and an image layout."""
    space = CayleySpace(permutation_group([[1, 0]]))
    assert space.space_kind == SpaceKind.LATTICE
    assert {'regular-geometry', 'image'} <= space.lacks
    assert not LatticeSpace(3).lacks


def test_group_space_numbers_sites_by_word_length():
    """Test that the words of the sites grow in length and that a ball is a prefix of a larger ball."""
    group = finitely_presented_group(['a', 'b'])
    small, large = CayleySpace(group, ball=2).prepare(), CayleySpace(group, ball=3).prepare()
    assert small.words_ == large.words_[: small.size_]
    assert list(large.lengths_) == sorted(large.lengths_)
    assert small.words_[:3] == ['1', 'a', 'a^-1']


def test_group_space_index_table_reads_outside_as_size():
    """Test that the identity reads itself, its generators the sites one letter away, and the boundary the fill."""
    space = CayleySpace(finitely_presented_group(['a']), ball=1).prepare()
    table = space._index_table(Memory(('1', 'a')))
    assert table[:, 0].tolist() == list(range(space.size_))
    assert table[0, 1] == space.words_.index('a')
    assert table[space.words_.index('a'), 1] == space.size_
    assert space.run(space.words_.index('a'), Memory(('1', 'a'))).tolist() == [space.words_.index('a')]
    assert sorted(space.run(0).tolist()) == [1, 2]


def test_group_space_index_table_needs_a_memory_set():
    """Test that a neighbourhood of offsets is rejected on a group space."""
    with pytest.raises(ValidationError, match='memory set'):
        CayleySpace(permutation_group([[1, 0]])).prepare().run(0, Neighborhood.radius())


def test_group_space_positions_lie_on_rings():
    """Test that every site sits at the distance of its word length from the identity at the origin."""
    space = CayleySpace(finitely_presented_group(['a', 'b']), ball=3).prepare()
    np.testing.assert_allclose(np.linalg.norm(space.positions_, axis=1), space.lengths_)


def test_group_space_edges_join_sites_by_generators():
    """Test that the edges of a cyclic group of order 4 form a cycle of one generator."""
    space = CayleySpace(permutation_group([[1, 2, 3, 0]])).prepare()
    assert sorted((first, second) for first, second, _ in space.edges_) == [(0, 1), (1, 3), (2, 0), (3, 2)]
    assert len(neighbor_pairs(space, space.size_)) == 4


def test_group_space_rejects_a_ball_and_a_quotient():
    """Test that a space is a ball or a quotient, not both."""
    with pytest.raises(ValidationError, match='either a ball or a quotient'):
        CayleySpace(permutation_group([[1, 0]]), ball=1, quotient=1).check()


def test_group_space_quotient_must_satisfy_the_relators():
    """Test that a quotient in which a relator is not the identity is rejected."""
    group = finitely_presented_group(['a'], ['a^3'])
    assert CayleySpace(group, quotient=permutation_group([[1, 2, 0]])).prepare().size_ == 3
    with pytest.raises(ValidationError, match='not a quotient'):
        CayleySpace(group, quotient=permutation_group([[1, 0]])).prepare()


def test_group_space_check_names_the_largest_radius_that_fits():
    """Test that a ball of the free group of rank 3 and radius 12 is refused, naming radius 6 as the largest."""
    with pytest.raises(SpaceTooLargeError, match='largest radius that fits is 6'):
        CayleySpace(finitely_presented_group(['a', 'b', 'c']), ball=12).check()


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_local_rule_free_group_majority(backend):
    """Test that every interior cell of a ball of radius 6 of the free group takes the majority of its five cells."""
    space = CayleySpace(finitely_presented_group(['a', 'b']), ball=6).prepare()
    cells = (np.random.default_rng(0).random(space.size_) < 0.5).astype(float)
    alive = _run(System(components=[CellComponent.lattice('cells', space, alive=cells)]), 1, backend)['cells.alive']
    table = space._index_table(FREE_MEMORY)
    interior = (table < space.size_).all(axis=1)
    expected = cells[table[interior]].sum(axis=1) >= 3
    np.testing.assert_array_equal(alive[1][interior], expected)


def test_local_rule_keeps_the_orientation_of_the_memory():
    """Test that on the Heisenberg group a cell copies the value at `g a`, not at `a g`."""
    space = CayleySpace(matrix_group(HEISENBERG), ball=3).prepare()
    elements = {space.words_[site]: site for site in range(space.size_)}
    cells = np.zeros(space.size_)
    cells[elements['b']] = 1.0
    alive = _run(System(components=[CopyComponent.lattice('cells', space, alive=cells)]), 1)['cells.alive'][1]
    assert alive[elements['b*a^-1']] == 1
    assert alive[elements['a^-1*b']] == 0


def test_local_rule_runs_without_gap(monkeypatch):
    """Test that a rule on a built ball of an automaton group steps many times with GAP unavailable."""
    space = CayleySpace(automaton_group(GRIGORCHUK), ball=4).prepare()
    cells = (np.random.default_rng(1).random(space.size_) < 0.5).astype(float)
    system = System(components=[CopyComponent.lattice('cells', space, alive=cells)])
    monkeypatch.setitem(sys.modules, 'sage.all__sagemath_gap', None)
    assert _run(system, 50)['cells.alive'].shape == (51, 40)


def test_local_rule_runs_on_a_level_quotient():
    """Test that the copy rule on the level-2 quotient of the Grigorchuk group is a permutation of the 8 cells."""
    space = CayleySpace(automaton_group(GRIGORCHUK), quotient=2).prepare()
    cells = np.arange(space.size_) % 2.0
    alive = _run(System(components=[CopyComponent.lattice('cells', space, alive=cells)]), 3)['cells.alive']
    assert space.size_ == 8
    assert (alive.sum(axis=1) == cells.sum()).all()
