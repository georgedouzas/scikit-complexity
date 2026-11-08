"""Test the `_cayley` module of `skcomplex.core.components.spaces`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.all__sagemath_symbolics import ZZ, matrix
from sage.groups.matrix_gps.finitely_generated import MatrixGroup
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import BoxNeighborhood, CayleySpace, MemoryNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import SpaceTooLargeError, ValidationError
from skcomplex.core.specs import StateSpec, TableSpec, TotalisticSpec, VisualSpec

HEISENBERG = (((1, 1, 0), (0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 1), (0, 0, 1)))
GRIGORCHUK = 'a = (1, 1)(1, 2), b = (a, c), c = (a, d), d = (1, b)'
MAJORITY = TotalisticSpec({3: 1, 4: 1, 5: 1})
FREE_MEMORY = MemoryNeighborhood(('1', 'a', 'a^-1', 'b', 'b^-1'))


def _grigorchuk():
    """Return the Grigorchuk group as an automaton group of AutomGrp."""
    libgap.LoadPackage('automgrp')
    return libgap.AutomatonGroup(GRIGORCHUK)


def _cyclic_presentation(order):
    """Return the cyclic group of an order as the finitely presented group `<a | a^order>`."""
    free = libgap.FreeGroup(['a'])
    (a,) = free.GeneratorsOfGroup()
    return free / [a**order]


class CellElement(Element):
    """Cell that takes the majority value of itself and its generators and their inverses."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive

    def _get_laws(self, s):
        """Return the majority rule."""
        return [LocalRuleLaw(s.alive, FREE_MEMORY, MAJORITY)]


class CopyElement(Element):
    """Cell that copies the value at the generator `a`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive

    def _get_laws(self, s):
        """Return the copy of the cell at `g a`."""
        return [LocalRuleLaw(s.alive, MemoryNeighborhood(('a',)), TableSpec({(1,): 1}))]


def _run(system, until):
    """Return the trajectory of a system."""
    return simulate(system, until=until, dt=1, random_state=0)


def test_cayley_space_checks_its_group_and_ball():
    """Test that a group that is not a Sage or GAP group and a negative radius are rejected by the check."""
    with pytest.raises(ValidationError, match='Sage or GAP group'):
        CayleySpace([[1, 0]]).prepare()
    with pytest.raises(ValidationError, match='nonnegative integer'):
        CayleySpace(libgap.FreeGroup(['a']), ball=-1).prepare()


def test_cayley_space_numbers_sites_by_word_length():
    """Test that the words of the sites grow in length and that a ball is a prefix of a larger ball."""
    group = libgap.FreeGroup(['a', 'b'])
    small, large = CayleySpace(group, ball=2).prepare(), CayleySpace(group, ball=3).prepare()
    assert small.words_ == large.words_[: small.size_]
    assert list(large.lengths_) == sorted(large.lengths_)
    assert small.words_[:3] == ['1', 'a', 'a^-1']
    assert small.ball_sizes_ == [1, 5, 17]


def test_cayley_space_index_table_reads_outside_as_size():
    """Test that the identity reads itself, its generators the sites one letter away, and the boundary the fill."""
    space = CayleySpace(libgap.FreeGroup(['a']), ball=1).prepare()
    table = space._index_table(MemoryNeighborhood(('1', 'a')))
    a = space.words_.index('a')
    assert table[:, 0].tolist() == list(range(space.size_))
    assert table[0, 1] == a
    assert table[a, 1] == space.size_
    assert space._get_neighbors(a, MemoryNeighborhood(('1', 'a'))).tolist() == [a]
    assert sorted(space._get_neighbors(0).tolist()) == [1, 2]


def test_cayley_space_index_table_needs_a_memory_set():
    """Test that a neighbourhood of offsets and a word of unknown generators are rejected on a Cayley space."""
    space = CayleySpace(PermutationGroup([[(1, 2)]])).prepare()
    with pytest.raises(ValidationError, match='memory set'):
        space._index_table(BoxNeighborhood())
    with pytest.raises(ValidationError, match='not a generator'):
        space._index_table(MemoryNeighborhood(('z',)))


def test_cayley_space_positions_lie_on_rings():
    """Test that every site sits at the distance of its word length from the identity at the origin."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=3).prepare()
    np.testing.assert_allclose(np.linalg.norm(space.positions_, axis=1), space.lengths_)


def test_cayley_space_edges_join_sites_by_generators():
    """Test that the edges of a cyclic group of order 4 form a cycle of one generator."""
    space = CayleySpace(PermutationGroup([[(1, 2, 3, 4)]])).prepare()
    assert sorted((first, second) for first, second, _ in space.edges_) == [(0, 1), (1, 3), (2, 0), (3, 2)]
    assert len(space._neighbor_pairs()) == 4
    assert (space.periodic_, space.default_visual_) == (True, 'cayley')


def test_cayley_space_solves_a_finitely_presented_group():
    """Test that the Cayley space of `<a | a^3>` holds the three elements of the cyclic group of order 3."""
    space = CayleySpace(_cyclic_presentation(3)).prepare()
    assert space.size_ == 3
    assert space.ball_sizes_ == [1, 3]


def test_cayley_space_solves_a_presentation_after_its_former_folder_is_removed(tmp_path, monkeypatch):
    """Test that Knuth-Bendix completion runs in the current folder, after the folder GAP last knew is removed."""
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    libgap.ChangeDirectoryCurrent(str(first))
    monkeypatch.chdir(second)
    first.rmdir()
    assert CayleySpace(_cyclic_presentation(4)).prepare().size_ == 4


def test_cayley_space_ball_of_a_finite_group_stops_at_the_group():
    """Test that a ball larger than a finite group holds every element once."""
    space = CayleySpace(PermutationGroup([[(1, 2)], [(1, 2, 3)]]), ball=3).prepare()
    assert (space.size_, space.ball_sizes_) == (6, [1, 4, 6, 6])


def test_cayley_space_refuses_more_sites_than_its_limit():
    """Test that a ball with more sites than the limit raises an error."""
    with pytest.raises(SpaceTooLargeError, match='more than 100 sites'):
        CayleySpace(libgap.FreeGroup(['a', 'b', 'c']), ball=6, max_sites=100).prepare()


def test_cayley_space_ball_is_the_memory_of_the_words_within_a_radius():
    """Test that the ball of a radius is the memory set of the words of the sites within it, the identity first."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2).prepare()
    assert space._ball(1).prepare().elements_ == tuple(space.words_[:5])


def test_local_rule_free_group_majority():
    """Test that every interior cell of a ball of radius 6 of the free group takes the majority of its five cells."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=6).prepare()
    cells = (np.random.default_rng(0).random(space.size_) < 0.5).astype(float)
    system = System(elements=[LatticeElement('cells', element=CellElement, space=space, values={'alive': cells})])
    alive = _run(system, 1).values('cells.alive')
    table = space._index_table(FREE_MEMORY)
    interior = (table < space.size_).all(axis=1)
    expected = cells[table[interior]].sum(axis=1) >= 3
    np.testing.assert_array_equal(alive[1][interior], expected)


def test_local_rule_keeps_the_orientation_of_the_memory():
    """Test that on the Heisenberg group a cell copies the value at `g a`, not at `a g`."""
    group = MatrixGroup([matrix(ZZ, generator) for generator in HEISENBERG])
    space = CayleySpace(group, ball=3).prepare()
    sites = {word: site for site, word in enumerate(space.words_)}
    cells = np.zeros(space.size_)
    cells[sites['b']] = 1.0
    system = System(elements=[LatticeElement('cells', element=CopyElement, space=space, values={'alive': cells})])
    alive = _run(system, 1).values('cells.alive')[1]
    assert alive[sites['b*a^-1']] == 1
    assert alive[sites['a^-1*b']] == 0


def test_local_rule_runs_on_a_ball_of_an_automaton_group():
    """Test that a rule on a ball of the Grigorchuk group of radius 4 steps on its 40 sites."""
    space = CayleySpace(_grigorchuk(), ball=4).prepare()
    cells = (np.random.default_rng(1).random(space.size_) < 0.5).astype(float)
    system = System(elements=[LatticeElement('cells', element=CopyElement, space=space, values={'alive': cells})])
    assert space.level_ > 0
    assert _run(system, 50).values('cells.alive').shape == (51, 40)


def test_local_rule_runs_on_a_level_quotient():
    """Test that the copy rule on the level-2 quotient of the Grigorchuk group is a permutation of the 8 cells."""
    space = CayleySpace(_grigorchuk().PermGroupOnLevel(2)).prepare()
    cells = np.arange(space.size_) % 2.0
    system = System(elements=[LatticeElement('cells', element=CopyElement, space=space, values={'alive': cells})])
    alive = _run(system, 3).values('cells.alive')
    assert space.size_ == 8
    assert (alive.sum(axis=1) == cells.sum()).all()
