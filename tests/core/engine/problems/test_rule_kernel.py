"""Test the `_rule_kernel` module of `skcomplex.core.engine.problems`."""

import itertools
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import (
    CayleySpace,
    LatticeNeighborhood,
    LatticeSpace,
    StateTranslationNeighborhood,
)
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import PowerSeriesSpec, StateSpec, TotalisticSpec
from skcomplex.systems.computer_science.automata import PowerSeriesCellElement

SYMMETRIC = PermutationGroup([[(1, 2)], [(1, 2, 3)], [(3, 4)]])
FORWARD = LatticeNeighborhood(((0,), (1,)))


class GroupValuedElement(Element):
    """Cell that holds a group element by the number of its site."""

    def __init__(self, label, *, value: Annotated[float, StateSpec(domain=ZZ)] = 0):
        super().__init__(label)
        self.value = value


def _translated(space, values, schedule='synchronous', rule=lambda value: value):
    """Return the system of a rule over the memory translated by the values of the cells."""
    return CompositeSystem(
        elements=[GroupElement('cells', element=GroupValuedElement, space=space, values={'value': values}, on='sites')],
        interactions=[
            LocalRuleInteraction(
                'shift',
                target='cells',
                state='value',
                rule=rule,
                neighborhood=StateTranslationNeighborhood(),
                schedule=schedule,
            ),
        ],
    )


def _values(system, until, schedule_random_state=0):
    """Return the values of the cells at every step of a run."""
    trajectory = simulate(system, until=until, dt=1, random_state=schedule_random_state)
    return trajectory.values('cells.value')


def test_translated_memory_reads_the_product_of_every_site_and_its_value():
    """Test that the compiled rule computes `x(g · x(g))` at every step on a finite group."""
    space = CayleySpace(SYMMETRIC).prepare()
    products = space._translation_table(StateTranslationNeighborhood())[:, :, 0]
    values = np.random.default_rng(1).integers(0, space.size_, space.size_).astype(float)
    trajectory = _values(_translated(space, values), 3)
    sites = np.arange(space.size_)
    for before, after in itertools.pairwise(trajectory):
        np.testing.assert_array_equal(after, before[products[sites, before.astype(int)]])


def test_translated_memory_commutes_with_translations_of_a_finite_group():
    """Test that translating a configuration by a group element translates its image, the equivariance of the map."""
    space = CayleySpace(SYMMETRIC).prepare()
    products = space._translation_table(StateTranslationNeighborhood())[:, :, 0]
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
    with pytest.raises(ValidationError, match='neither degrees nor a coloring'):
        simulate(_translated(space, np.zeros(space.size_), schedule='colored'), until=0)


def test_function_rule_updates_a_power_series_state():
    """Test that `x(n) - t x(n + 1)` acts on rows of coefficients, as the binomial coefficients of `(1 - t sigma)^s`."""
    ring = PowerSeriesSpec(4)
    values = np.zeros((6, 4))
    values[3, 0] = 1.0

    def difference(here, right):
        return here - ring.multiply(ring.coefficients_of('t'), right)

    system = CompositeSystem(
        elements=[
            GroupElement(
                'cells',
                element=PowerSeriesCellElement,
                space=LatticeSpace(6),
                values={'ring': ring, 'value': values},
                on='sites',
            ),
        ],
        interactions=[
            LocalRuleInteraction('difference', target='cells', state='value', rule=difference, neighborhood=FORWARD),
        ],
    )
    final = _values(system, 3)[-1]
    expected = np.zeros((6, 4))
    expected[3, 0], expected[2, 1], expected[1, 2], expected[0, 3] = 1, -3, 3, -1
    np.testing.assert_array_equal(final, expected)


def test_lookup_rules_reject_power_series_states():
    """Test that a totalistic rule cannot read a state of several components."""
    system = CompositeSystem(
        elements=[GroupElement('cells', element=PowerSeriesCellElement, space=LatticeSpace(5), on='sites')],
        interactions=[
            LocalRuleInteraction(
                'sum',
                target='cells',
                state='value',
                rule=TotalisticSpec({1: 1}),
                neighborhood=FORWARD,
            ),
        ],
    )
    with pytest.raises(ValidationError, match='one component'):
        simulate(system, until=0)
