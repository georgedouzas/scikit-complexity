"""Test the `_cayley` module of `skcomplex.systems.computer_science.common.automata`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.all__sagemath_symbolics import ZZ, matrix
from sage.groups.matrix_gps.finitely_generated import MatrixGroup
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TableSpec, TotalisticSpec, UpdateLaw
from skcomplex.core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from skcomplex.core.engine.problems import FrameSpec
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import SpaceError, SpaceTooLargeError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.computer_science.common.automata import (
    CayleySpace,
    CayleyVisual,
    MemoryNeighborhood,
    StateTranslationNeighborhood,
)
from skcomplex.systems.computer_science.common.automata import _cayley as cayley

HEISENBERG = (((1, 1, 0), (0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 1), (0, 0, 1)))
GRIGORCHUK = 'a = (1, 1)(1, 2), b = (a, c), c = (a, d), d = (1, b)'
MAJORITY = TotalisticSpec({3: 1, 4: 1, 5: 1})
FREE_MEMORY = MemoryNeighborhood(('1', 'a', 'a^-1', 'b', 'b^-1'))
SYMMETRIC = PermutationGroup([[(1, 2)], [(1, 2, 3)]])
COPY = {'rule': TableSpec({(1,): 1}), 'neighborhood': MemoryNeighborhood(('a',))}
PARITY = {'rule': TotalisticSpec({1: 1, 3: 1}), 'neighborhood': MemoryNeighborhood(('1', 'a', 'b'))}


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
    """Binary cells on the sites of a space."""

    def __init__(
        self,
        label: str,
        *,
        alive: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.alive = alive


class LocalRuleInteraction(Interaction):
    """Local rule of the cells, a table or a totalistic rule over a memory set."""

    def __init__(self, label: str, *, target: Role = 'cells', rule=MAJORITY, neighborhood=FREE_MEMORY) -> None:
        super().__init__(label)
        self.target = target
        self.rule = rule
        self.neighborhood = neighborhood

    def _prepare(self, context=None):
        """Set the local rule of every cell."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target.alive, self.rule, neighborhood=self.neighborhood)]


def _cells(space, cells, **rule):
    """Return a system of binary cells on a space with a local rule."""
    return System(
        elements=[CellElement('cells', alive=cells, space=space)],
        interactions=[LocalRuleInteraction('rule', **rule)],
    )


def _run(system, until):
    """Return the trajectory of a system."""
    return system.prepare().simulate(until=until, dt=1, random_state=0)


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
    table = space._get_index_table(MemoryNeighborhood(('1', 'a')))
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
        space._get_index_table(LatticeNeighborhood(((-1,), (0,), (1,))))
    with pytest.raises(ValidationError, match='not a generator'):
        space._get_index_table(MemoryNeighborhood(('z',)))


def test_cayley_space_positions_lie_on_rings():
    """Test that every site sits at the distance of its word length from the identity at the origin."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=3).prepare()
    np.testing.assert_allclose(np.linalg.norm(space.positions_, axis=1), space.lengths_)


def test_cayley_space_edges_join_sites_by_generators():
    """Test that the edges of a cyclic group of order 4 form a cycle of one generator."""
    space = CayleySpace(PermutationGroup([[(1, 2, 3, 4)]])).prepare()
    assert sorted((first, second) for first, second, _ in space.edges_) == [(0, 1), (1, 3), (2, 0), (3, 2)]
    assert len(space._get_neighbor_pairs()) == 4
    assert (space.periodic_, space.default_visual_) == (True, CayleyVisual)


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
    assert space._get_ball(1).prepare().elements_ == tuple(space.words_[:5])


def test_local_rule_free_group_majority():
    """Test that every interior cell of a ball of radius 6 of the free group takes the majority of its five cells."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=6).prepare()
    cells = (np.random.default_rng(0).random(space.size_) < 0.5).astype(float)
    system = _cells(space, cells)
    alive = _run(system, 1).get_values('cells.alive')
    table = space._get_index_table(FREE_MEMORY)
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
    system = _cells(space, cells, **COPY)
    alive = _run(system, 1).get_values('cells.alive')[1]
    assert alive[sites['b*a^-1']] == 1
    assert alive[sites['a^-1*b']] == 0


def test_local_rule_runs_on_a_ball_of_an_automaton_group():
    """Test that a rule on a ball of the Grigorchuk group of radius 4 steps on its 40 sites."""
    space = CayleySpace(_grigorchuk(), ball=4).prepare()
    cells = (np.random.default_rng(1).random(space.size_) < 0.5).astype(float)
    system = _cells(space, cells, **COPY)
    assert space.level_ > 0
    assert _run(system, 50).get_values('cells.alive').shape == (51, 40)


def test_local_rule_runs_on_a_level_quotient():
    """Test that the copy rule on the level-2 quotient of the Grigorchuk group is a permutation of the 8 cells."""
    space = CayleySpace(_grigorchuk().PermGroupOnLevel(2)).prepare()
    cells = np.arange(space.size_) % 2.0
    system = _cells(space, cells, **COPY)
    alive = _run(system, 3).get_values('cells.alive')
    assert space.size_ == 8
    assert (alive.sum(axis=1) == cells.sum()).all()


def test_memory_neighbourhood_keeps_its_words_in_order():
    """Test that a memory set prepares its words as strings in the given order."""
    assert MemoryNeighborhood(['a*b', '1', 'a^-1']).prepare().elements_ == ('a*b', '1', 'a^-1')


def test_neighborhood_keeps_its_words():
    """Test that the neighbourhood holds the words read after the translation, the identity by default."""
    assert StateTranslationNeighborhood().prepare().elements_ == ('1',)
    assert StateTranslationNeighborhood(['1', 'a']).prepare().elements_ == ('1', 'a')


def test_translation_table_of_a_finite_group_is_its_cayley_table():
    """Test that the products of a finite group have the identity, are associative, and permute every row."""
    space = CayleySpace(SYMMETRIC).prepare()
    table = space._get_translation_table(StateTranslationNeighborhood())[:, :, 0]
    sites = np.arange(space.size_)
    np.testing.assert_array_equal(table[:, 0], sites)
    np.testing.assert_array_equal(table[0], sites)
    np.testing.assert_array_equal(np.sort(table, axis=1), np.broadcast_to(sites, table.shape))
    for first in sites:
        for second in sites:
            np.testing.assert_array_equal(table[table[first, second]], table[first][table[second]])
    letter = space.words_.index('a')
    np.testing.assert_array_equal(table[:, letter], space.generator_table_[:, 0])


def test_translation_table_reads_words_after_the_product():
    """Test that a word of the memory set multiplies the product on the right."""
    space = CayleySpace(SYMMETRIC).prepare()
    table = space._get_translation_table(StateTranslationNeighborhood(('1', 'a')))
    letter = space.generator_table_[:, 0]
    np.testing.assert_array_equal(table[:, :, 1], letter[table[:, :, 0]])


def test_translation_table_of_a_ball_leaves_it_outside():
    """Test that a product outside a ball of the free group is the number of sites."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2).prepare()
    table = space._get_translation_table(StateTranslationNeighborhood())[:, :, 0]
    lengths = space.lengths_
    inside = lengths[:, None] + lengths[None, :] <= 2
    assert np.all(table[inside] < space.size_)
    assert np.all((table == space.size_) | (lengths[np.minimum(table, space.size_ - 1)] <= 2))
    assert (table == space.size_).any()


def test_translation_table_rejects_other_neighborhoods_spaces_and_large_tables(monkeypatch):
    """Test that the hook takes a translated memory on a group space within the size limit."""
    space = CayleySpace(SYMMETRIC).prepare()
    with pytest.raises(ValidationError, match='translates a memory set'):
        space._get_translation_table(MemoryNeighborhood(('1',)))
    with pytest.raises(SpaceError, match='no group space'):
        LatticeSpace(5).prepare()._get_translation_table(StateTranslationNeighborhood())
    monkeypatch.setattr(cayley, 'MAX_PRODUCTS', 20)
    with pytest.raises(SpaceTooLargeError, match='finite quotient'):
        space._get_translation_table(StateTranslationNeighborhood())


class Logs:
    """Recording stand-in that keeps the entity paths, archetypes, and static flags it receives."""

    def __init__(self):
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity."""
        self.logs.append((path, archetype, static))


def _trajectory():
    """Return the trajectory of the parity rule on a ball of radius 3 of the free group of rank 2."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=3).prepare()
    cells = np.zeros(space.size_)
    cells[0] = 1.0
    return _cells(space, cells, **PARITY).prepare().simulate(until=2)


def test_cayley_visual_logs_nodes_on_rings_and_edges_once():
    """Test that the nodes sit on rings by word length, colored by state, and the edges are logged once as static."""
    trajectory = _trajectory()
    problem = trajectory.problem
    logs = Logs()
    visual = CayleyVisual(path=problem.elements[0].path, recording=logs).prepare(problem)
    for _ in range(2):
        visual.run(FrameSpec(0, 0.0, trajectory.states[-1], trajectory.parameters[-1], problem))
    nodes = [archetype for path, archetype, static in logs.logs if not static]
    edges = [archetype for path, archetype, static in logs.logs if static]
    assert len(nodes) == 2
    assert len(edges) == 1
    positions = np.asarray(nodes[0].positions.as_arrow_array().to_pylist())
    space = problem.elements[0].element.space
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1) / 40.0, space.lengths_)
    assert len(edges[0].edges.as_arrow_array().to_pylist()) == len(space.edges_)


def test_cayley_visual_views_are_graph_views():
    """Test that cells on a group space default to the Cayley visual, whose view is a graph view of their state."""
    trajectory = _trajectory()
    problem = trajectory.problem
    views = CayleyVisual(path=problem.elements[0].path).prepare(problem).views_
    assert [(type(view).__name__, str(view.origin)) for view in views] == [('GraphView', 'cells/alive')]
    assert problem.elements[0].visual.kind is CayleyVisual
    assert RerunObserver(blueprint='default').prepare(problem).blueprint_ is not None
