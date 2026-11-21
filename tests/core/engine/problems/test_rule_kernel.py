"""Test the `_rule_kernel` module of `skcomplex.core.engine.problems`."""

import itertools
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components import PowerSeriesSpec
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import TotalisticSpec, UpdateLaw
from skcomplex.core.components.spaces import (
    BaseSpace,
    LatticeNeighborhood,
    LatticeSpace,
)
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.computer_science.automata import PowerSeriesCellElement
from skcomplex.systems.computer_science.common.automata import CayleySpace, StateTranslationNeighborhood

SYMMETRIC = PermutationGroup([[(1, 2)], [(1, 2, 3)], [(3, 4)]])
FORWARD = LatticeNeighborhood(((0,), (1,)))


class GroupValuedElement(Element):
    """Cell that holds a group element by the number of its site."""

    def __init__(self, label, *, value: Annotated[int, ZZ] = 0, space: BaseSpace | None = None):
        super().__init__(label, space=space)
        self.value = value


def _local_rule(label, rule, neighborhood, schedule=None):
    """Return the interaction of a local rule of the value of the cells over a neighbourhood."""
    return Interaction(
        label,
        roles={'target': 'cells'},
        laws=lambda s: [UpdateLaw(s.target.value, rule, neighborhood=neighborhood, schedule=schedule)],
    )


def _translated(space, values, schedule='synchronous', rule=lambda value, **kwargs: value):
    """Return the system of a rule over the memory translated by the values of the cells."""
    return System(
        elements=[GroupValuedElement('cells', value=values, space=space)],
        interactions=[_local_rule('shift', rule, StateTranslationNeighborhood(), schedule)],
    )


def _values(system, until, schedule_random_state=0):
    """Return the values of the cells at every step of a run."""
    trajectory = system.prepare().simulate(until=until, dt=1, random_state=schedule_random_state)
    return trajectory.get_values('cells.value')


def test_translated_memory_reads_the_product_of_every_site_and_its_value():
    """Test that the compiled rule computes `x(g · x(g))` at every step on a finite group."""
    space = CayleySpace(SYMMETRIC).prepare()
    products = space._get_translation_table(StateTranslationNeighborhood())[:, :, 0]
    values = np.random.default_rng(1).integers(0, space.size_, space.size_).astype(float)
    trajectory = _values(_translated(space, values), 3)
    sites = np.arange(space.size_)
    for before, after in itertools.pairwise(trajectory):
        np.testing.assert_array_equal(after, before[products[sites, before.astype(int)]])


def test_translated_memory_commutes_with_translations_of_a_finite_group():
    """Test that translating a configuration by a group element translates its image, the equivariance of the map."""
    space = CayleySpace(SYMMETRIC).prepare()
    products = space._get_translation_table(StateTranslationNeighborhood())[:, :, 0]
    values = np.random.default_rng(2).integers(0, space.size_, space.size_).astype(float)
    element = 5
    inverse = int(np.flatnonzero(products[element] == 0)[0])
    translated = values[products[inverse]]
    image = _values(_translated(space, values), 1)[-1]
    np.testing.assert_array_equal(_values(_translated(space, translated), 1)[-1], image[products[inverse]])


def test_translated_memory_reads_the_fill_on_a_ball():
    """Test that a product outside a ball of the free group reads the fill value, which then stays."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2, fill=-1).prepare()
    values = np.full(space.size_, space.size_ - 1, dtype=float)
    trajectory = _values(_translated(space, values), 2)
    assert trajectory[1, 0] == space.size_ - 1
    assert (trajectory[1] == -1).sum() > 0
    assert set(np.unique(trajectory[2][trajectory[1] == -1])) == {-1}


def test_translated_memory_gives_no_degree():
    """Test that a rule over a translated memory receives `degree` as `None`."""
    degrees = []

    def keep(value, *, degree, **kwargs):
        degrees.append(degree)
        return value

    space = CayleySpace(SYMMETRIC).prepare()
    _values(_translated(space, np.zeros(space.size_), rule=keep), 1)
    assert degrees == [None]


def test_translated_memory_runs_random_sequentially():
    """Test that a random sequential schedule updates every site once from the latest values."""
    space = CayleySpace(SYMMETRIC).prepare()
    values = np.zeros(space.size_)
    values[0] = 3
    trajectory = _values(_translated(space, values, schedule='random-sequential'), 1)
    assert set(np.unique(trajectory[-1])) <= set(np.unique(values))


def test_translated_memory_rejects_colorings_and_lookups_of_series():
    """Test that a colored schedule, which needs a fixed table, is rejected with a translated memory."""
    space = CayleySpace(SYMMETRIC).prepare()
    with pytest.raises(ValidationError, match='has no coloring'):
        _translated(space, np.zeros(space.size_), schedule='colored').prepare().simulate(until=0)


def test_function_rule_updates_a_power_series_state():
    """Test that `x(n) - t x(n + 1)` acts on rows of coefficients, as the binomial coefficients of `(1 - t sigma)^s`."""
    ring = PowerSeriesSpec(4)
    values = np.zeros((6, 4))
    values[3, 0] = 1.0

    def difference(here, right, **kwargs):
        return here - ring.multiply(ring.get_coefficients_of('t'), right)

    cells = PowerSeriesCellElement('cells', ring=ring, value=values, space=LatticeSpace(6))
    rule = Interaction(
        'difference',
        roles={'target': 'cells'},
        laws=lambda s: [UpdateLaw(s.target.value, difference, neighborhood=FORWARD)],
    )
    system = System(elements=[cells], interactions=[rule]).prepare()
    final = system.simulate(until=3, dt=1, random_state=0).get_values('cells.value')[-1]
    expected = np.zeros((6, 4))
    expected[3, 0], expected[2, 1], expected[1, 2], expected[0, 3] = 1, -3, 3, -1
    np.testing.assert_array_equal(final, expected)


def test_lookup_rules_reject_power_series_states():
    """Test that a totalistic rule cannot read a state of several components."""
    cells = PowerSeriesCellElement('cells', space=LatticeSpace(5))
    rule = Interaction(
        'sum',
        roles={'target': 'cells'},
        laws=lambda s: [UpdateLaw(s.target.value, TotalisticSpec({1: 1}), neighborhood=FORWARD)],
    )
    with pytest.raises(ValidationError, match='one component'):
        System(elements=[cells], interactions=[rule]).prepare().simulate(until=1)
