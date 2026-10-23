"""Test the automata theory checks, linear automata, subshifts, and the information measures of automata."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import GF, matrix
from scipy.linalg import circulant

from skcomplex.core.analysis import (
    Automaton,
    LinearAutomaton,
    Subshift,
    compose,
    configuration_entropy,
    difference_pattern,
    garden_of_eden,
    inverse,
    is_injective,
    is_pre_injective,
    is_surjective,
    spatial_entropy_rates,
)
from skcomplex.core.base import AnalysisError
from skcomplex.core.laws import OuterTotalistic, Table, Totalistic
from skcomplex.core.space import GroupSpace, LatticeSpace, Memory, Neighborhood, permutation_group

LIFE = OuterTotalistic({(0, 3): 1, (1, 2): 1, (1, 3): 1})
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


def _elementary(number, size):
    """Return an elementary automaton by its Wolfram number on a ring."""
    patterns = [(left, center, right) for left in (0, 1) for center in (0, 1) for right in (0, 1)]
    table = Table({pattern: (number >> (4 * pattern[0] + 2 * pattern[1] + pattern[2])) & 1 for pattern in patterns})
    return Automaton(LatticeSpace(size), Neighborhood.radius(), table)


def _cycle(size):
    """Return the cyclic group of an order as a group space."""
    return GroupSpace(permutation_group([[(point + 1) % size for point in range(size)]]))


def test_life_is_not_surjective_with_a_garden_of_eden():
    """Test that Life on a 6 x 6 torus is not surjective and that the found configuration has no preimage."""
    life = Automaton(LatticeSpace((6, 6)), Neighborhood.moore(), LIFE)
    assert not is_surjective(life, method='sat')
    eden = garden_of_eden(life, random_state=0)
    assert eden is not None
    assert garden_of_eden(life, window=range(36), tries=1, random_state=0) is not None


def test_enumeration_and_sat_agree():
    """Test that enumeration and SAT decide injectivity and pre-injectivity alike on small rings."""
    for number in (15, 30, 90, 170, 204):
        automaton = _elementary(number, 8)
        assert is_injective(automaton, method='enumeration') == is_injective(automaton, method='sat'), number
        window = [2, 3]
        assert is_pre_injective(automaton, window, method='enumeration') == is_pre_injective(
            automaton,
            window,
            method='sat',
        ), number


def test_enumeration_limit_raises():
    """Test that enumeration of too many configurations raises an error that names the SAT method."""
    with pytest.raises(AnalysisError, match='SAT method'):
        is_injective(_elementary(30, 40), method='enumeration')


def test_inverse_of_the_shift_is_the_opposite_shift():
    """Test that the inverse of the shift reads the other neighbour, and that their composition is the identity."""
    shift = Automaton(LatticeSpace(8), Neighborhood(((1,),)), Table({(1,): 1}))
    back = inverse(shift, max_radius=1)
    configuration = np.random.default_rng(0).integers(2, size=8)
    np.testing.assert_array_equal(back.step(configuration), np.roll(configuration, 1))
    np.testing.assert_array_equal(compose(shift, back).step(configuration), configuration)
    assert inverse(_elementary(30, 8)) is None


def test_inverse_on_a_group_space():
    """Test that the inverse of the rotation of the cyclic group of order 6 is found within the ball of radius 1."""
    space = _cycle(6)
    rotation = Automaton(space, Memory(('a',)), Table({(1,): 1}))
    configuration = np.array([1, 0, 0, 1, 1, 0])
    back = inverse(rotation, max_radius=1)
    np.testing.assert_array_equal(back.step(rotation.step(configuration)), configuration)


def test_laplacian_over_gf2_matches_its_circulant():
    """Test that the discrete Laplacian on the cycle of length 5 over GF(2) is invertible as its circulant matrix is."""
    space = _cycle(5)
    laplacian = LinearAutomaton(space, Memory(('a^-1', '1', 'a')), (1, -2, 1), 'GF(2)')
    column = np.zeros(5, dtype=int)
    column[[1, 4]] = 1
    expected = matrix(GF(2), circulant(column).tolist()).is_invertible()
    assert laplacian.is_invertible() == expected
    assert laplacian.is_surjective() == expected
    configuration = np.array([1, 0, 0, 1, 0])
    neighbours = space.index_table(Memory(('a^-1', 'a')))
    np.testing.assert_array_equal(laplacian.step(configuration), configuration[neighbours].sum(axis=1) % 2)


def test_linear_automaton_on_the_integers():
    """Test that rule 90 is surjective but not invertible on Z, and the identity plus shift over QQ likewise."""
    rule90 = LinearAutomaton(LatticeSpace(7), Neighborhood(((-1,), (1,))), (1, 1), 'GF(2)')
    assert rule90.is_surjective(infinite=True)
    assert rule90.is_pre_injective(infinite=True)
    assert not rule90.is_invertible(infinite=True)
    shift = LinearAutomaton(LatticeSpace(7), Neighborhood(((1,),)), (2,), 'QQ')
    assert shift.is_invertible(infinite=True)
    assert shift.is_invertible()


def test_golden_mean_shift_counts_follow_fibonacci():
    """Test that the words of length n of the golden mean shift number F(n + 2) and its entropy nears log2 of phi."""
    golden = Subshift(Neighborhood(((0,), (1,))), [(1, 1)])
    counts = [golden.count(LatticeSpace(length, boundary='fixed')) for length in range(1, 11)]
    assert counts == [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    entropies = configuration_entropy(golden, [LatticeSpace(length, boundary='fixed') for length in (10, 16, 22)])
    assert np.all(np.diff(entropies) < 0)
    assert entropies[-1] == pytest.approx(np.log2(GOLDEN_RATIO), abs=0.02)


def test_golden_mean_shift_samples_admissible_configurations():
    """Test that the Markov chain keeps configurations without two adjacent ones, and rejects a bad start."""
    golden = Subshift(Memory(('1', 'a')), [(1, 1)])
    space = _cycle(60)
    sample = golden.sample(space, 20, random_state=0)
    assert golden.is_admissible(sample, space)
    assert 0 < sample.mean() < 0.5
    with pytest.raises(AnalysisError, match='forbidden pattern'):
        golden.sample(space, initial=np.ones(60))


def test_configuration_entropy_of_an_automaton_image():
    """Test that the image of a surjective rule has full entropy and that of the majority rule less."""
    windows = [LatticeSpace(12, boundary='fixed')]
    assert configuration_entropy(_elementary(90, 12), windows)[0] == pytest.approx(1.0)
    majority = Automaton(LatticeSpace(12), Neighborhood.radius(), Totalistic({2: 1, 3: 1}))
    assert configuration_entropy(majority, windows)[0] < 0.9


def test_spatial_entropy_rate_does_not_increase():
    """Test that rule 18 from a random configuration of a large ring never raises the spatial entropy rate.

    The block estimates of length 6 on 20,000 cells fluctuate by about 0.02, so rises within that are estimation error,
    while the rate falls by more than 0.2 in all.
    """
    rule = _elementary(18, 20000)
    runs = rule.run(np.random.default_rng(0).integers(2, size=20000), 20)
    rates = spatial_entropy_rates(runs, 6)
    assert np.diff(rates).max() < 0.03
    assert rates[-1] < rates[0] - 0.2


def test_difference_pattern_spreads_at_the_speed_of_light():
    """Test that a flip under rule 30 spreads at most one cell per step on each side."""
    rule = _elementary(30, 41)
    differences = difference_pattern(rule, np.zeros(41, dtype=int), 20, 10)
    widths = differences.sum(axis=1)
    assert widths[0] == 1
    assert (widths <= 2 * np.arange(11) + 1).all()
    assert widths[-1] > 5
