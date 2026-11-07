"""Test the `_alphabets` module of `skcomplex.systems.computer_science.automata`."""

from math import comb

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup

from skcomplex.core.components.spaces import StateTranslationNeighborhood
from skcomplex.core.errors import ParameterError
from skcomplex.systems.computer_science.automata import (
    GroupDifferenceAutomatonSystem,
    PowerSeriesAutomatonSystem,
    RealQuadraticAutomatonSystem,
    SelfIndexedShiftSystem,
    SubtractionAutomatonSystem,
)

from ..conftest import _run


def test_self_indexed_shift_reads_the_product_of_every_site_and_its_value():
    """Test that every cell takes the value at the product of its site and its value, on a finite group."""
    system = (
        SelfIndexedShiftSystem(PermutationGroup([[(1, 2)], [(1, 2, 3, 4)]]), radius=None, random_state=3)
        .prepare()
        .system_
    )
    space = system.elements[0].space.prepare()
    products = space._translation_table(StateTranslationNeighborhood())[:, :, 0]
    values = _run(system, 2).values('cells.value').astype(int)
    sites = np.arange(space.size_)
    np.testing.assert_array_equal(values[1], values[0][products[sites, values[0]]])
    np.testing.assert_array_equal(values[2], values[1][products[sites, values[1]]])


def test_self_indexed_shift_loses_the_products_outside_a_ball():
    """Test that on a ball of the free group the cells whose product leaves the ball hold no element, and keep it."""
    values = _run(SelfIndexedShiftSystem(radius=3), 4).values('cells.value')
    lost = (values == -1).mean(axis=1)
    assert lost[0] == 0
    assert np.all(np.diff(lost) >= 0)
    assert lost[-1] > 0.5


def test_power_series_automaton_spreads_binomial_coefficients():
    """Test that from the series 1 the cell `k` to the left holds `C(s, k) (-t)^k` after `s` steps."""
    steps, order = 5, 8
    values = _run(PowerSeriesAutomatonSystem(21, order=order), steps).values('cells.value')[-1]
    middle = 10
    for k in range(steps + 1):
        expected = np.zeros(order)
        expected[k] = comb(steps, k) * (-1) ** k
        np.testing.assert_array_equal(values[middle - k], expected)


def test_power_series_automaton_is_inverted_by_the_geometric_series():
    """Test that `sigma(y)(n) = sum_j t^j y(n + j)` recovers the configuration, the inverse of Exercise 1.37."""
    rng = np.random.default_rng(0)
    order, size = 5, 9
    initial = rng.integers(-2, 3, (size, order)).astype(float)
    image = _run(PowerSeriesAutomatonSystem(size, order=order, initial=initial), 1).values('cells.value')[-1]
    recovered = np.zeros_like(image)
    for shift in range(order):
        recovered[:, shift:] += np.roll(image, -shift, axis=0)[:, : order - shift]
    np.testing.assert_array_equal(recovered, initial)


def test_power_series_automaton_over_a_prime_field_stays_reduced():
    """Test that over the field of three elements every coefficient stays a residue."""
    values = _run(PowerSeriesAutomatonSystem(15, order=6, field='GF(3)'), 10).values('cells.value')
    assert set(np.unique(values)) <= {0.0, 1.0, 2.0}


def test_subtraction_automaton_maps_the_staircase_to_a_block_of_ones():
    """Test that `τ(x_M)` is the block of `M` ones from the origin, the sequence that closes on no image."""
    for height in (3, 6, 9):
        values = _run(SubtractionAutomatonSystem(40, height=height), 1).values('cells.value')[-1]
        expected = np.zeros(40)
        expected[10 : 10 + height] = 1
        np.testing.assert_array_equal(values, expected)
    with pytest.raises(ParameterError, match='natural number'):
        SubtractionAutomatonSystem(height=-1).prepare()


def test_real_quadratic_automaton_maps_zero_and_one_to_zero():
    """Test that the constant configurations 0 and 1 have the same image, so the automaton is not injective."""
    zero = _run(RealQuadraticAutomatonSystem(8, initial=np.zeros(8)), 1).values('cells.value')[-1]
    one = _run(RealQuadraticAutomatonSystem(8, initial=np.ones(8)), 1).values('cells.value')[-1]
    np.testing.assert_array_equal(zero, one)
    values = _run(RealQuadraticAutomatonSystem(16), 1).values('cells.value')
    np.testing.assert_allclose(values[1], np.roll(values[0], -1) - values[0] ** 2)


def test_group_difference_automaton_maps_constants_to_the_identity():
    """Test that `x(n)^-1 x(n + 1)` sends every constant configuration to the identity and multiplies in the group."""
    for element in range(6):
        values = _run(GroupDifferenceAutomatonSystem(7, initial=np.full(7, element)), 1).values('cells.value')[-1]
        np.testing.assert_array_equal(values, 0)
    system = GroupDifferenceAutomatonSystem(30, random_state=1).prepare().system_
    values = _run(system, 1).values('cells.value').astype(int)
    rule = system.interactions[0].rule
    product, inverse = rule.keywords['product'], rule.keywords['inverse']
    np.testing.assert_array_equal(values[1], product[inverse[values[0]], np.roll(values[0], -1)])
    ordered = product[np.arange(6), inverse]
    np.testing.assert_array_equal(ordered, 0)
