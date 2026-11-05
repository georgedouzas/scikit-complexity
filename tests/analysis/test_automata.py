"""Test the automaton analysis: decisions, inverses, compositions, difference patterns, and entropies."""

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.all__sagemath_symbolics import GF, matrix
from scipy.linalg import circulant

from skcomplex.analysis import AutomatonAnalysis, EntropyAnalysis
from skcomplex.core.spaces import (
    BoxNeighborhood,
    CayleySpace,
    LatticeNeighborhood,
    LatticeSpace,
    MemoryNeighborhood,
    MooreNeighborhood,
)
from skcomplex.errors import AnalysisError
from skcomplex.models.computer_science.automata import Automaton, LinearAutomaton, Subshift, lamplighter
from skcomplex.specs import OuterTotalisticSpec, TableSpec, TotalisticSpec

LIFE = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
QUOTIENT_EDEN = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 26, 28, 29, 30]
LIFE_EDEN = [*QUOTIENT_EDEN, 33, 35]


def _automaton(kind, source, **settings):
    """Return the automaton analysis of a model for one kind, run."""
    return AutomatonAnalysis(kind, **settings).check().prepare(source).run()


def _elementary(number, size):
    """Return an elementary automaton by its Wolfram number on a ring."""
    patterns = [(left, center, right) for left in (0, 1) for center in (0, 1) for right in (0, 1)]
    table = TableSpec({pattern: (number >> (4 * pattern[0] + 2 * pattern[1] + pattern[2])) & 1 for pattern in patterns})
    return Automaton(LatticeSpace(size), BoxNeighborhood(), table)


def _cycle(size):
    """Return the cyclic group of an order as a group space."""
    return CayleySpace(PermutationGroup([[tuple(range(1, size + 1))]]))


def test_life_is_not_surjective_with_a_garden_of_eden():
    """Test that Life on a 6 x 6 torus is not surjective and that the found configuration has no preimage."""
    life = Automaton(LatticeSpace((6, 6)), MooreNeighborhood(), LIFE)
    assert not _automaton('surjective', life, method='sat').surjective_
    eden = _automaton('garden-of-eden', life, random_state=0).pattern_
    assert eden is not None
    assert _automaton('garden-of-eden', life, window=range(36), tries=1, random_state=0).pattern_ is not None


def test_enumeration_and_sat_agree():
    """Test that enumeration and SAT decide injectivity and pre-injectivity alike on small rings."""
    for number in (15, 30, 90, 170, 204):
        automaton = _elementary(number, 8)
        enumerated = _automaton('injective', automaton, method='enumeration').injective_
        assert enumerated == _automaton('injective', automaton, method='sat').injective_, number
        window = [2, 3]
        enumerated = _automaton('pre-injective', automaton, window=window, method='enumeration').pre_injective_
        assert enumerated == _automaton('pre-injective', automaton, window=window, method='sat').pre_injective_, number


def test_enumeration_limit_raises():
    """Test that enumeration of too many configurations raises an error that names the SAT method."""
    with pytest.raises(AnalysisError, match='SAT method'):
        _automaton('injective', _elementary(30, 40), method='enumeration')


def test_inverse_of_the_shift_is_the_opposite_shift():
    """Test that the inverse of the shift reads the other neighbour, and that their composition is the identity."""
    shift = Automaton(LatticeSpace(8), LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
    back = _automaton('inverse', shift, max_radius=1).inverse_
    configuration = np.random.default_rng(0).integers(2, size=8)
    np.testing.assert_array_equal(back.step(configuration), np.roll(configuration, 1))
    composition = _automaton('compose', shift, second=back).composition_
    np.testing.assert_array_equal(composition.step(configuration), configuration)
    assert _automaton('inverse', _elementary(30, 8)).inverse_ is None


def test_inverse_on_a_group_space():
    """Test that the inverse of the rotation of the cyclic group of order 6 is found within the ball of radius 1."""
    space = _cycle(6)
    rotation = Automaton(space, MemoryNeighborhood(('a',)), TableSpec({(1,): 1}))
    configuration = np.array([1, 0, 0, 1, 1, 0])
    back = _automaton('inverse', rotation, max_radius=1).inverse_
    np.testing.assert_array_equal(back.step(rotation.step(configuration)), configuration)


def test_laplacian_over_gf2_matches_its_circulant():
    """Test that the discrete Laplacian on the cycle of length 5 over GF(2) is invertible as its circulant matrix is."""
    space = _cycle(5)
    laplacian = LinearAutomaton(space, MemoryNeighborhood(('a^-1', '1', 'a')), (1, -2, 1), 'GF(2)')
    column = np.zeros(5, dtype=int)
    column[[1, 4]] = 1
    expected = matrix(GF(2), circulant(column).tolist()).is_invertible()
    assert _automaton('injective', laplacian).injective_ == expected
    assert _automaton('surjective', laplacian).surjective_ == expected
    configuration = np.array([1, 0, 0, 1, 0])
    neighbours = space.prepare()._index_table(MemoryNeighborhood(('a^-1', 'a')))
    np.testing.assert_array_equal(laplacian.step(configuration), configuration[neighbours].sum(axis=1) % 2)


def test_linear_automaton_on_the_integers():
    """Test that rule 90 is surjective but not invertible on Z, and the identity plus shift over QQ likewise."""
    rule90 = LinearAutomaton(LatticeSpace(7), LatticeNeighborhood(((-1,), (1,))), (1, 1), 'GF(2)')
    assert _automaton('surjective', rule90, infinite=True).surjective_
    assert _automaton('pre-injective', rule90, infinite=True).pre_injective_
    assert not _automaton('injective', rule90, infinite=True).injective_
    shift = LinearAutomaton(LatticeSpace(7), LatticeNeighborhood(((1,),)), (2,), 'QQ')
    assert _automaton('injective', shift, infinite=True).injective_
    assert _automaton('injective', shift).injective_


def test_golden_mean_shift_entropy_nears_log_phi():
    """Test that the entropy of the golden mean shift on growing windows decreases towards log2 of phi."""
    golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
    windows = [LatticeSpace(length, boundary='fixed') for length in (10, 16, 22)]
    entropies = _automaton('entropy', golden, windows=windows).entropy_
    assert np.all(np.diff(entropies) < 0)
    assert entropies[-1] == pytest.approx(np.log2(GOLDEN_RATIO), abs=0.02)


def test_configuration_entropy_of_an_automaton_image():
    """Test that the image of a surjective rule has full entropy and that of the majority rule less."""
    windows = [LatticeSpace(12, boundary='fixed')]
    assert _automaton('entropy', _elementary(90, 12), windows=windows).entropy_[0] == pytest.approx(1.0)
    majority = Automaton(LatticeSpace(12), BoxNeighborhood(), TotalisticSpec({2: 1, 3: 1}))
    assert _automaton('entropy', majority, windows=windows).entropy_[0] < 0.9


def test_spatial_entropy_rate_does_not_increase():
    """Test that rule 18 from a random configuration of a large ring never raises the spatial entropy rate.

    The block estimates of length 6 on 20,000 cells fluctuate by about 0.02, so rises within that are estimation error,
    while the rate falls by more than 0.2 in all.
    """
    rule = _elementary(18, 20000)
    runs = rule.run(np.random.default_rng(0).integers(2, size=20000), 20)
    rates = EntropyAnalysis('rate', dimension=6).check().prepare(runs.T).run().entropy_
    assert np.diff(rates).max() < 0.03
    assert rates[-1] < rates[0] - 0.2


def test_difference_pattern_spreads_at_the_speed_of_light():
    """Test that a flip under rule 30 spreads at most one cell per step on each side."""
    rule = _elementary(30, 41)
    differences = _automaton('difference', rule, configuration=np.zeros(41, dtype=int), site=20, steps=10).differences_
    widths = differences.sum(axis=1)
    assert widths[0] == 1
    assert (widths <= 2 * np.arange(11) + 1).all()
    assert widths[-1] > 5


def test_kind_reads_its_model():
    """Test that a subshift is rejected by a decision, and a setting another kind reads by the check."""
    golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
    with pytest.raises(AnalysisError, match='reads an automaton or a linear automaton'):
        AutomatonAnalysis().check().prepare(golden)
    with pytest.raises(AnalysisError, match='does not read tries'):
        AutomatonAnalysis(tries=3).check()


def test_reference_values_on_a_lattice():
    """Test that the decisions, gardens of Eden, inverses, and compositions on lattices keep their recorded values."""
    life = Automaton(LatticeSpace((6, 6)), MooreNeighborhood(), LIFE)
    assert not _automaton('surjective', life).surjective_
    assert not _automaton('injective', life).injective_
    assert np.flatnonzero(_automaton('garden-of-eden', life, random_state=0).pattern_).tolist() == LIFE_EDEN
    majority = Automaton(LatticeSpace(5), BoxNeighborhood(), TotalisticSpec(lambda total: total >= 2))
    assert _automaton('garden-of-eden', majority).pattern_.tolist() == [0, 0, 1, 0, 1]
    assert _automaton('garden-of-eden', _elementary(90, 8)).pattern_.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    decisions = {
        number: (
            _automaton('injective', _elementary(number, 8)).injective_,
            _automaton('pre-injective', _elementary(number, 8), window=[2, 3]).pre_injective_,
        )
        for number in (15, 30, 90, 170, 204)
    }
    assert decisions == {
        15: (True, True),
        30: (False, True),
        90: (False, True),
        170: (True, True),
        204: (True, True),
    }
    shift = Automaton(LatticeSpace(8), LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
    assert _automaton('inverse', _elementary(30, 8)).inverse_ is None
    back = _automaton('inverse', shift).inverse_
    assert back.reads.check().prepare().offsets_ == ((-1,), (0,), (1,))
    assert sorted(pattern for pattern, value in back.rule.mapping.items() if value) == [
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    composition = _automaton('compose', shift, second=back).composition_
    assert composition.reads.check().prepare().offsets_ == ((0,), (1,), (2,))
    assert composition.local_map().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert composition.step([1, 0, 1, 1, 0, 0, 0, 1]).tolist() == [1, 0, 1, 1, 0, 0, 0, 1]


def test_reference_values_on_a_group():
    """Test that the kinds on a cyclic group and on a quotient of the lamplighter group keep their recorded values."""
    rotation = Automaton(_cycle(6), MemoryNeighborhood(('a',)), TableSpec({(1,): 1}))
    assert rotation.step([1, 0, 0, 1, 1, 0]).tolist() == [0, 1, 1, 0, 0, 1]
    assert rotation.table().ravel().tolist() == [1, 3, 0, 5, 2, 4]
    assert _automaton('injective', rotation).injective_
    assert _automaton('surjective', rotation).surjective_
    assert _automaton('pre-injective', rotation, window=[1, 2]).pre_injective_
    assert _automaton('garden-of-eden', rotation, tries=5, random_state=0).pattern_ is None
    back = _automaton('inverse', rotation, max_radius=1).inverse_
    assert back.reads.elements == ('1', 'a', 'a^-1')
    assert sorted(pattern for pattern, value in back.rule.mapping.items() if value) == [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ]
    quotient = CayleySpace(libgap.PermGroupOnLevel(lamplighter(), 3))
    memory = MemoryNeighborhood(('1', 'a', 'b'))
    parity = Automaton(quotient, memory, TotalisticSpec(lambda total: total % 2))
    assert _automaton('injective', parity, method='sat').injective_
    assert _automaton('surjective', parity, method='sat').surjective_
    majority = Automaton(quotient, memory, TotalisticSpec(lambda total: total >= 2))
    assert not _automaton('injective', majority).injective_
    assert not _automaton('pre-injective', majority, window=[0, 1]).pre_injective_
    assert np.flatnonzero(_automaton('garden-of-eden', majority, random_state=0).pattern_).tolist() == QUOTIENT_EDEN
    linear = LinearAutomaton(quotient, memory, (1, 1, 1), 'GF(2)')
    assert linear.matrix().rank() == 32
    assert _automaton('injective', linear).injective_
    assert _automaton('surjective', linear).surjective_


def test_reference_values_of_linear_decisions_and_measures():
    """Test that the linear decisions, the difference pattern, and the entropies keep their recorded values."""
    rule90 = LinearAutomaton(LatticeSpace(6), LatticeNeighborhood(((-1,), (1,))), (1, 1), 'GF(2)')
    assert not _automaton('injective', rule90).injective_
    assert not _automaton('surjective', rule90).surjective_
    assert not _automaton('injective', rule90, infinite=True).injective_
    assert _automaton('surjective', rule90, infinite=True).surjective_
    assert _automaton('pre-injective', rule90, infinite=True).pre_injective_
    laplacian = LinearAutomaton(_cycle(5), MemoryNeighborhood(('a^-1', '1', 'a')), (1, -2, 1), 'GF(2)')
    assert laplacian.step([1, 0, 0, 1, 0]).tolist() == [0, 0, 1, 0, 1]
    assert not _automaton('pre-injective', laplacian).pre_injective_
    settings = {'configuration': np.zeros(41, dtype=int), 'site': 20, 'steps': 10}
    differences = _automaton('difference', _elementary(30, 41), **settings).differences_
    assert differences.sum(axis=1).tolist() == [1, 3, 3, 6, 4, 9, 5, 12, 7, 12, 11]
    golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
    windows = [LatticeSpace(length, boundary='fixed') for length in (10, 20)]
    entropies = _automaton('entropy', golden, windows=windows).entropy_
    assert entropies.tolist() == pytest.approx([0.7169925001442313, 0.7056179025755023])
    natural = _automaton('entropy', golden, windows=windows[1:], base=np.e).entropy_
    assert natural.tolist() == pytest.approx([0.4890970597228316])
    majority = Automaton(LatticeSpace(12), BoxNeighborhood(), TotalisticSpec({2: 1, 3: 1}))
    image = _automaton('entropy', majority, windows=[LatticeSpace(12, boundary='fixed')]).entropy_
    assert image.tolist() == pytest.approx([0.8954196310386876])
