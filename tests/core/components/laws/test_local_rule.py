"""Test the `_local_rule` module of `skcomplex.core.components.laws`."""

import itertools

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import (
    AdjacencyNeighborhood,
    BallNeighborhood,
    CayleySpace,
    ContinuousTime,
    DiscreteTime,
    LatticeNeighborhood,
    LatticeSpace,
    MemoryNeighborhood,
    StateTranslationNeighborhood,
)
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import LinearRuleSpec, OuterTotalisticSpec, TableSpec, TotalisticSpec
from skcomplex.systems.computer_science.automata import Wolfram

LIFE = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})


def test_local_rule_targets_one_state_or_several_together():
    """Test that a local rule targets one lattice state, or a tuple of them that it updates together."""
    height, grains = var('height grains')
    assert LocalRuleLaw(
        height,
        BallNeighborhood(n_dim=1, center='inline'),
        TotalisticSpec({2: 1}),
    ).prepare().targets_ == (height,)
    together = LocalRuleLaw(
        (height, grains),
        BallNeighborhood(n_dim=1, center='inline'),
        lambda z, *n: (z, z),
    ).prepare()
    assert together.targets_ == (height, grains)


def test_local_rule_requires_discrete_time_on_sites():
    """Test that a local rule runs in discrete time on a lattice, a network, or a Cayley graph."""
    requirements = LocalRuleLaw(
        var('cell'),
        BallNeighborhood(n_dim=1, center='inline'),
        TotalisticSpec({2: 1}),
    )._get_requirements()
    assert requirements.time == (DiscreteTime,)
    assert LatticeSpace in requirements.space
    assert CayleySpace in requirements.space
    assert ContinuousTime not in requirements.time


def test_local_rule_rejects_an_unknown_schedule():
    """Test that a local rule rejects a schedule it does not know."""
    with pytest.raises(ValidationError, match='Unknown schedule'):
        LocalRuleLaw(
            var('cell'),
            BallNeighborhood(n_dim=1, center='inline'),
            TotalisticSpec({2: 1}),
            schedule='parallel',
        ).prepare()


def _run(rule, neighborhood, space, configuration):
    """Return one step of a local rule on a space."""
    law = LocalRuleLaw(var('alive'), neighborhood, rule).prepare()
    return law.run({'space': space, 'alive': configuration}).value_


def _local_map(rule, neighborhood, space, site, states=2):
    """Return the next value at a site of every pattern of the cells it reads, by one run on every configuration."""
    size = space.prepare().size_
    patterns = list(itertools.product(range(states), repeat=size))
    return _run(rule, neighborhood, space, np.asarray(patterns))[:, site]


def test_run_shift_moves_every_cell_one_site():
    """Test that the shift reads the right neighbour, so every step rolls the ring to the left."""
    configuration = np.array([1, 0, 0, 1, 1, 0])
    image = _run(TableSpec({(1,): 1}), LatticeNeighborhood(((1,),)), LatticeSpace(6), configuration)
    np.testing.assert_array_equal(image, np.roll(configuration, -1))


def test_run_reads_a_rule_code_through_its_table():
    """Test that a Wolfram code gives rule 90, the sum of the two neighbours modulo 2."""
    image = _run(Wolfram(90), BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(7), [0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(image, [0, 0, 1, 0, 1, 0, 0])


def test_run_outer_totalistic_rule_reads_the_cell_first():
    """Test that Life turns a blinker and keeps the recorded steps of a glider."""
    space, around = LatticeSpace((5, 5)), BallNeighborhood()
    blinker = np.zeros((5, 5), dtype=int)
    blinker[2, 1:4] = 1
    np.testing.assert_array_equal(_run(LIFE, around, space, blinker.ravel()).reshape(5, 5), blinker.T)
    glider = np.zeros((5, 5), dtype=int)
    glider[[0, 1, 2, 2, 2], [1, 2, 0, 1, 2]] = 1
    current = glider.ravel()
    assert np.flatnonzero(_run(LIFE, around, space, current)).tolist() == [5, 7, 11, 12, 16]
    for _ in range(4):
        current = _run(LIFE, around, space, current)
    assert np.flatnonzero(current).tolist() == [7, 13, 16, 17, 18]


def test_run_keeps_the_values_of_elementary_rules():
    """Test that rules 90 and 30 keep their recorded runs and local maps, on one row and on a batch of rows."""
    space = LatticeSpace(11)
    current = np.eye(11, dtype=int)[5]
    rows = [current]
    for _ in range(4):
        current = _run(Wolfram(90), BallNeighborhood(n_dim=1, center='inline'), space, current)
        rows.append(current)
    assert [np.flatnonzero(row).tolist() for row in rows] == [[5], [4, 6], [3, 7], [2, 4, 6, 8], [1, 9]]
    assert _local_map(Wolfram(90), BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(3), 1).tolist() == [
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
    ]
    batch = _run(Wolfram(90), BallNeighborhood(n_dim=1, center='inline'), space, np.eye(11, dtype=int)[:2])
    assert [np.flatnonzero(row).tolist() for row in batch] == [[1, 10], [0, 2]]
    assert _local_map(Wolfram(30), BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(3), 1).tolist() == [
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
    ]
    current = np.eye(41, dtype=int)[20]
    for _ in range(10):
        current = _run(Wolfram(30), BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(41), current)
    assert np.flatnonzero(current).tolist() == [10, 11, 14, 19, 21, 22, 23, 24, 26, 27, 30]


def test_run_keeps_the_values_of_other_rules():
    """Test that fixed edges with a fill, totalistic, function, and three-state rules keep their recorded steps."""
    fixed = LatticeSpace(5, boundary='fixed', fill=1)
    assert _run(Wolfram(90), BallNeighborhood(n_dim=1, center='inline'), fixed, [0, 0, 1, 0, 0]).tolist() == [
        1,
        1,
        0,
        1,
        1,
    ]
    parity = TotalisticSpec(lambda total: total % 2)
    assert _run(parity, BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(5), [0, 0, 1, 0, 0]).tolist() == [
        0,
        1,
        1,
        1,
        0,
    ]

    def outer_sum(left, _, right):
        """Return the sum of the outer neighbours modulo 2."""
        return (left + right) % 2

    assert _run(outer_sum, BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(5), [0, 0, 1, 0, 0]).tolist() == [
        0,
        1,
        0,
        1,
        0,
    ]
    three = TotalisticSpec(lambda total: total % 3, states=3)
    assert _run(three, BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(5), [0, 2, 1, 0, 0]).tolist() == [
        2,
        0,
        0,
        1,
        0,
    ]
    pair = LatticeNeighborhood(((0,), (1,)))
    assert _local_map(three, pair, LatticeSpace(2), 0, states=3).tolist() == [0, 1, 2, 1, 2, 0, 2, 0, 1]
    mapping = TotalisticSpec({1: 1}, default=0)
    assert _run(mapping, BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(5), [0, 0, 1, 0, 0]).tolist() == [
        0,
        1,
        1,
        1,
        0,
    ]
    outer = OuterTotalisticSpec(lambda center, total: (center + total) % 2)
    assert _run(outer, LatticeNeighborhood(((1,),)), LatticeSpace(3), [1, 0, 0]).tolist() == [1, 0, 1]


def test_run_keeps_the_values_of_linear_rules():
    """Test that linear rules keep their recorded steps over a prime field and the rationals."""
    around = LatticeNeighborhood(((-1,), (1,)))
    rule90 = LinearRuleSpec((1, 1), 'GF(2)')
    assert _run(rule90, around, LatticeSpace(6), [0, 0, 1, 0, 0, 0]).tolist() == [0, 1, 0, 1, 0, 0]
    assert _run(rule90, around, LatticeSpace(6), [1, 0, 1, 1, 0, 0]).tolist() == [0, 0, 1, 1, 1, 1]
    laplacian = LinearRuleSpec((1, -2, 1), 'QQ')
    assert _run(laplacian, BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(5), [1, 0, 0, 1, 0]).tolist() == [
        -2,
        1,
        1,
        -2,
        2,
    ]
    fractions = LinearRuleSpec((0.5, 0.25), 'QQ')
    pair = LatticeNeighborhood(((0,), (1,)))
    assert _run(fractions, pair, LatticeSpace(3), [1, 2, 4]).tolist() == [1.0, 2.0, 2.25]
    modular = LinearRuleSpec((1, -2, 1), 'GF(3)')
    assert _run(modular, BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(5), [1, 0, 0, 2, 0]).tolist() == [
        1,
        1,
        2,
        2,
        0,
    ]


def test_run_function_rule_reads_keywords_and_sets_several_targets():
    """Test that a function rule reads the keywords of the input, and returns one array per target."""
    height, grains = var('height grains')
    own = LatticeNeighborhood(((0,),))
    reading = LocalRuleLaw((height, grains), own, lambda x, y: (x + y, x - y), reads=(height, grains))
    first, second = reading.prepare().run({'space': LatticeSpace(2), 'height': [1, 2], 'grains': [3, 5]}).value_
    assert first.tolist() == [4, 7]
    assert second.tolist() == [-2, -3]
    scaled = LocalRuleLaw(var('cell'), LatticeNeighborhood(((0,),)), lambda x, *, scale: x * scale).prepare()
    assert scaled.run({'space': LatticeSpace(2), 'cell': [1, 2], 'scale': 0.5}).value_.tolist() == [0.5, 1.0]


def test_run_rejects_invalid_inputs_and_rules():
    """Test that a run without the space or a state, or with an unknown rule or coefficients, raises an error."""
    law = LocalRuleLaw(var('cell'), BallNeighborhood(n_dim=1, center='inline'), Wolfram(90)).prepare()
    with pytest.raises(ValidationError, match='runs on a mapping'):
        law.run({'cell': [0, 1, 0]})
    with pytest.raises(ValidationError, match='runs on a mapping'):
        law.run({'space': LatticeSpace(3)})
    with pytest.raises(ValidationError, match='is not a TableSpec'):
        _run(object(), BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(3), [0, 1, 0])
    with pytest.raises(ValidationError, match='not a linear rule of 3 neighbours'):
        _run(LinearRuleSpec((1,)), BallNeighborhood(n_dim=1, center='inline'), LatticeSpace(3), [0, 1, 0])
    with pytest.raises(ValidationError, match='Unknown field'):
        LocalRuleLaw(
            var('cell'),
            BallNeighborhood(n_dim=1, center='inline'),
            LinearRuleSpec((1, 1, 1), 'GF(x)'),
        ).prepare()
    graph = AdjacencyNeighborhood()
    with pytest.raises(ValidationError, match='on offsets or a memory set'):
        _run(LIFE, graph, LatticeSpace(3), [0, 1, 0])


def test_run_outer_rule_on_a_memory_set_reads_the_identity_first():
    """Test that an outer totalistic rule on a group space reads the site itself before its memory."""
    space = CayleySpace(PermutationGroup([[(1, 2, 3)]]))
    rule = OuterTotalisticSpec({(1, 0): 1, (0, 1): 1})
    assert sorted(_run(rule, MemoryNeighborhood(('a',)), space, [1, 0, 0]).tolist()) == [0, 1, 1]


def test_run_translated_memory_reads_the_product_of_a_site_and_its_value():
    """Test that `x(g · x(g))` reads the product of every site and its value, row by row of configurations."""
    space = CayleySpace(PermutationGroup([[(1, 2)], [(1, 2, 3)]])).prepare()
    products = space._get_translation_table(StateTranslationNeighborhood())[:, :, 0]
    configurations = np.random.default_rng(0).integers(0, space.size_, (4, space.size_))
    images = _run(lambda value: value, StateTranslationNeighborhood(), space, configurations)
    sites = np.arange(space.size_)
    for configuration, image in zip(configurations, images, strict=True):
        np.testing.assert_array_equal(image, configuration[products[sites, configuration]])


def test_run_translated_memory_reads_the_fill_outside_a_ball_or_for_no_element():
    """Test that a product outside a ball, or a value that names no site, reads the fill value."""
    space = CayleySpace(PermutationGroup([[(1, 2, 3, 4, 5, 6)]]), ball=1, fill=-1).prepare()
    image = _run(lambda value: value, StateTranslationNeighborhood(), space, [1, 1, -1])
    assert image.tolist() == [1, -1, -1]
