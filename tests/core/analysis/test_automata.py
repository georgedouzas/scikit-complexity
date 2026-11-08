"""Test the automaton analysis: decisions, inverses, compositions, difference patterns, entropies, and subshifts."""

import hashlib
import itertools
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.all__sagemath_symbolics import GF, matrix, var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from scipy.linalg import circulant

from skcomplex.core.analysis import automaton, entropy
from skcomplex.core.components.elements import DeclaredElement, Element, LatticeElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import (
    BoxNeighborhood,
    CayleySpace,
    LatticeNeighborhood,
    LatticeSpace,
    MemoryNeighborhood,
    MooreNeighborhood,
)
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import (
    DeclaredVariableSpec,
    LinearRuleSpec,
    OuterTotalisticSpec,
    StateSpec,
    SubshiftSpec,
    TableSpec,
    TotalisticSpec,
)
from skcomplex.systems.computer_science.automata import lamplighter

LIFE = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
QUOTIENT_EDEN = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 26, 28, 29, 30]
LIFE_EDEN = [*QUOTIENT_EDEN, 33, 35]
GOLDEN_PAIR = LatticeNeighborhood(((0,), (1,)))


class CellElement(Element):
    """Cell of an automaton with the values 0 and 1."""

    def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive


def _automaton_system(space, neighborhood, rule, **settings):
    """Return the system of an automaton of a space, a neighbourhood, and a rule."""
    cells = LatticeElement('cells', element=CellElement, space=space)
    interaction = LocalRuleInteraction(
        'rule',
        target='cells',
        state='alive',
        rule=rule,
        neighborhood=neighborhood,
        **settings,
    )
    return System(elements=[cells], interactions=[interaction])


def _subshift_system(space, neighborhood, forbidden, states=2):
    """Return the system of a group of cells whose state has a subshift as its domain."""
    domain = SubshiftSpec(neighborhood, forbidden, states)
    variables = {'alive': DeclaredVariableSpec(0, StateSpec(domain=domain))}
    return System(
        elements=[LatticeElement('cells', element=DeclaredElement, space=space, values={'variables': variables})],
    )


def _automaton(kind, source, **settings):
    """Return the decision or measure of an automaton for one kind."""
    return automaton(source, kind=kind, **settings)


def _law(system):
    """Return the law of the local rule of a system."""
    interaction = system.interactions[0]
    return LocalRuleLaw(var('alive'), interaction.neighborhood, interaction.rule).prepare()


def _step(system, configuration):
    """Return the image of a configuration under one run of the law of a system."""
    inputs = {'space': system.elements[0].space, 'alive': np.asarray(configuration)}
    return _law(system).run(inputs).value_


def _elementary(number, size):
    """Return an elementary automaton by its Wolfram number on a ring."""
    patterns = [(left, center, right) for left in (0, 1) for center in (0, 1) for right in (0, 1)]
    table = TableSpec({pattern: (number >> (4 * pattern[0] + 2 * pattern[1] + pattern[2])) & 1 for pattern in patterns})
    return _automaton_system(LatticeSpace(size), BoxNeighborhood(), table)


def _cycle(size):
    """Return the cyclic group of an order as a group space."""
    return CayleySpace(PermutationGroup([[tuple(range(1, size + 1))]]))


def test_life_is_not_surjective_with_a_garden_of_eden():
    """Test that Life on a 6 x 6 torus is not surjective and that the found configuration has no preimage."""
    life = _automaton_system(LatticeSpace((6, 6)), MooreNeighborhood(), LIFE)
    assert not _automaton('surjective', life, method='sat').surjective
    eden = _automaton('garden-of-eden', life, random_state=0).pattern
    assert eden is not None
    assert _automaton('garden-of-eden', life, window=range(36), tries=1, random_state=0).pattern is not None


def test_enumeration_and_sat_agree():
    """Test that enumeration and SAT decide injectivity and pre-injectivity alike on small rings."""
    for number in (15, 30, 90, 170, 204):
        automaton = _elementary(number, 8)
        enumerated = _automaton('injective', automaton, method='enumeration').injective
        assert enumerated == _automaton('injective', automaton, method='sat').injective, number
        window = [2, 3]
        enumerated = _automaton('pre-injective', automaton, window=window, method='enumeration').pre_injective
        assert enumerated == _automaton('pre-injective', automaton, window=window, method='sat').pre_injective, number


def test_enumeration_limit_raises():
    """Test that enumeration of too many configurations raises an error that names the SAT method."""
    with pytest.raises(AnalysisError, match='SAT method'):
        _automaton('injective', _elementary(30, 40), method='enumeration')


def test_inverse_of_the_shift_is_the_opposite_shift():
    """Test that the inverse of the shift reads the other neighbour, and that their composition is the identity."""
    shift = _automaton_system(LatticeSpace(8), LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
    back = _automaton('inverse', shift, max_radius=1).inverse
    configuration = np.random.default_rng(0).integers(2, size=8)
    np.testing.assert_array_equal(_step(back, configuration), np.roll(configuration, 1))
    composition = _automaton('compose', shift, second=back).composition
    np.testing.assert_array_equal(_step(composition, configuration), configuration)
    assert _automaton('inverse', _elementary(30, 8)).inverse is None


def test_inverse_on_a_group_space():
    """Test that the inverse of the rotation of the cyclic group of order 6 is found within the ball of radius 1."""
    rotation = _automaton_system(_cycle(6), MemoryNeighborhood(('a',)), TableSpec({(1,): 1}))
    configuration = np.array([1, 0, 0, 1, 1, 0])
    back = _automaton('inverse', rotation, max_radius=1).inverse
    np.testing.assert_array_equal(_step(back, _step(rotation, configuration)), configuration)


def test_laplacian_over_gf2_matches_its_circulant():
    """Test that the discrete Laplacian on the cycle of length 5 over GF(2) is invertible as its circulant matrix is."""
    space = _cycle(5)
    laplacian = _automaton_system(space, MemoryNeighborhood(('a^-1', '1', 'a')), LinearRuleSpec((1, -2, 1), 'GF(2)'))
    column = np.zeros(5, dtype=int)
    column[[1, 4]] = 1
    expected = matrix(GF(2), circulant(column).tolist()).is_invertible()
    assert _automaton('injective', laplacian).injective == expected
    assert _automaton('surjective', laplacian).surjective == expected
    configuration = np.array([1, 0, 0, 1, 0])
    neighbours = space.prepare()._index_table(MemoryNeighborhood(('a^-1', 'a')))
    np.testing.assert_array_equal(_step(laplacian, configuration), configuration[neighbours].sum(axis=1) % 2)


def test_linear_automaton_on_the_integers():
    """Test that rule 90 is surjective but not invertible on Z, and the identity plus shift over QQ likewise."""
    rule90 = _automaton_system(LatticeSpace(7), LatticeNeighborhood(((-1,), (1,))), LinearRuleSpec((1, 1), 'GF(2)'))
    assert _automaton('surjective', rule90, infinite=True).surjective
    assert _automaton('pre-injective', rule90, infinite=True).pre_injective
    assert not _automaton('injective', rule90, infinite=True).injective
    shift = _automaton_system(LatticeSpace(7), LatticeNeighborhood(((1,),)), LinearRuleSpec((2,), 'QQ'))
    assert _automaton('injective', shift, infinite=True).injective
    assert _automaton('injective', shift).injective


def test_golden_mean_shift_entropy_nears_log_phi():
    """Test that the entropy of the golden mean shift on growing windows decreases towards log2 of phi."""
    golden = _subshift_system(LatticeSpace(10, boundary='fixed'), GOLDEN_PAIR, ((1, 1),))
    windows = [LatticeSpace(length, boundary='fixed') for length in (10, 16, 22)]
    entropies = _automaton('entropy', golden, windows=windows).entropy
    assert np.all(np.diff(entropies) < 0)
    assert entropies[-1] == pytest.approx(np.log2(GOLDEN_RATIO), abs=0.02)


def test_configuration_entropy_of_an_automaton_image():
    """Test that the image of a surjective rule has full entropy and that of the majority rule less."""
    windows = [LatticeSpace(12, boundary='fixed')]
    assert _automaton('entropy', _elementary(90, 12), windows=windows).entropy[0] == pytest.approx(1.0)
    majority = _automaton_system(LatticeSpace(12), BoxNeighborhood(), TotalisticSpec({2: 1, 3: 1}))
    assert _automaton('entropy', majority, windows=windows).entropy[0] < 0.9


def test_spatial_entropy_rate_does_not_increase():
    """Test that rule 18 from a random configuration of a large ring never raises the spatial entropy rate.

    The block estimates of length 6 on 20,000 cells fluctuate by about 0.02, so rises within that are estimation error,
    while the rate falls by more than 0.2 in all.
    """
    rule = _elementary(18, 20000)
    rows = [np.random.default_rng(0).integers(2, size=20000)]
    for _ in range(20):
        rows.append(_step(rule, rows[-1]))
    rates = entropy(np.stack(rows).T, kind='rate', dimension=6).entropy
    assert np.diff(rates).max() < 0.03
    assert rates[-1] < rates[0] - 0.2


def test_difference_pattern_spreads_at_the_speed_of_light():
    """Test that a flip under rule 30 spreads at most one cell per step on each side."""
    rule = _elementary(30, 41)
    differences = _automaton('difference', rule, configuration=np.zeros(41, dtype=int), site=20, steps=10).differences
    widths = differences.sum(axis=1)
    assert widths[0] == 1
    assert (widths <= 2 * np.arange(11) + 1).all()
    assert widths[-1] > 5


def test_kind_reads_its_system():
    """Test that a subshift is rejected by a decision, and a setting another kind reads by the check."""
    golden = _subshift_system(LatticeSpace(5), GOLDEN_PAIR, ((1, 1),))
    with pytest.raises(AnalysisError, match='reads a system with an automaton or a linear automaton'):
        automaton(golden)
    with pytest.raises(AnalysisError, match='does not read tries'):
        automaton(golden, tries=3)
    with pytest.raises(AnalysisError, match='reads a system with a subshift, not ndarray'):
        automaton(np.zeros(3), kind='count')
    with pytest.raises(AnalysisError, match='needs configuration'):
        automaton(golden, kind='admissible')


def test_reference_values_on_a_lattice():
    """Test that the decisions, gardens of Eden, inverses, and compositions on lattices keep their recorded values."""
    life = _automaton_system(LatticeSpace((6, 6)), MooreNeighborhood(), LIFE)
    assert not _automaton('surjective', life).surjective
    assert not _automaton('injective', life).injective
    assert np.flatnonzero(_automaton('garden-of-eden', life, random_state=0).pattern).tolist() == LIFE_EDEN
    majority = _automaton_system(LatticeSpace(5), BoxNeighborhood(), TotalisticSpec(lambda total: total >= 2))
    assert _automaton('garden-of-eden', majority).pattern.tolist() == [0, 0, 1, 0, 1]
    assert _automaton('garden-of-eden', _elementary(90, 8)).pattern.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    decisions = {
        number: (
            _automaton('injective', _elementary(number, 8)).injective,
            _automaton('pre-injective', _elementary(number, 8), window=[2, 3]).pre_injective,
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
    shift = _automaton_system(LatticeSpace(8), LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
    assert _automaton('inverse', _elementary(30, 8)).inverse is None
    back = _automaton('inverse', shift).inverse
    assert back.interactions[0].neighborhood.prepare().offsets_ == ((-1,), (0,), (1,))
    assert back.interactions[0].rule.n_states == 2
    assert sorted(pattern for pattern, value in back.interactions[0].rule.mapping.items() if value) == [
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    composition = _automaton('compose', shift, second=back).composition
    assert composition.interactions[0].neighborhood.prepare().offsets_ == ((0,), (1,), (2,))
    configurations = np.zeros((8, 8), dtype=int)
    configurations[:, :3] = list(itertools.product((0, 1), repeat=3))
    assert _step(composition, configurations)[:, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert _step(composition, [1, 0, 1, 1, 0, 0, 0, 1]).tolist() == [1, 0, 1, 1, 0, 0, 0, 1]


def test_reference_values_on_a_group():
    """Test that the kinds on a cyclic group and on a quotient of the lamplighter group keep their recorded values."""
    space = _cycle(6)
    rotation = _automaton_system(space, MemoryNeighborhood(('a',)), TableSpec({(1,): 1}))
    assert _step(rotation, [1, 0, 0, 1, 1, 0]).tolist() == [0, 1, 1, 0, 0, 1]
    assert space.prepare()._index_table(MemoryNeighborhood(('a',))).ravel().tolist() == [1, 3, 0, 5, 2, 4]
    assert _automaton('injective', rotation).injective
    assert _automaton('surjective', rotation).surjective
    assert _automaton('pre-injective', rotation, window=[1, 2]).pre_injective
    assert _automaton('garden-of-eden', rotation, tries=5, random_state=0).pattern is None
    back = _automaton('inverse', rotation, max_radius=1).inverse
    assert back.interactions[0].neighborhood.elements == ('1', 'a', 'a^-1')
    assert sorted(pattern for pattern, value in back.interactions[0].rule.mapping.items() if value) == [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ]
    quotient = CayleySpace(libgap.PermGroupOnLevel(lamplighter(), 3))
    memory = MemoryNeighborhood(('1', 'a', 'b'))
    parity = _automaton_system(quotient, memory, TotalisticSpec(lambda total: total % 2))
    assert _automaton('injective', parity, method='sat').injective
    assert _automaton('surjective', parity, method='sat').surjective
    majority = _automaton_system(quotient, memory, TotalisticSpec(lambda total: total >= 2))
    assert not _automaton('injective', majority).injective
    assert not _automaton('pre-injective', majority, window=[0, 1]).pre_injective
    assert np.flatnonzero(_automaton('garden-of-eden', majority, random_state=0).pattern).tolist() == QUOTIENT_EDEN
    linear = _automaton_system(quotient, memory, LinearRuleSpec((1, 1, 1), 'GF(2)'))
    assert _automaton('matrix', linear).matrix.rank() == 32
    assert _automaton('injective', linear).injective
    assert _automaton('surjective', linear).surjective


def test_reference_values_of_linear_decisions_and_measures():
    """Test that the linear decisions, the difference pattern, and the entropies keep their recorded values."""
    rule90 = _automaton_system(LatticeSpace(6), LatticeNeighborhood(((-1,), (1,))), LinearRuleSpec((1, 1), 'GF(2)'))
    assert not _automaton('injective', rule90).injective
    assert not _automaton('surjective', rule90).surjective
    assert not _automaton('injective', rule90, infinite=True).injective
    assert _automaton('surjective', rule90, infinite=True).surjective
    assert _automaton('pre-injective', rule90, infinite=True).pre_injective
    memory = MemoryNeighborhood(('a^-1', '1', 'a'))
    laplacian = _automaton_system(_cycle(5), memory, LinearRuleSpec((1, -2, 1), 'GF(2)'))
    assert _step(laplacian, [1, 0, 0, 1, 0]).tolist() == [0, 0, 1, 0, 1]
    assert not _automaton('pre-injective', laplacian).pre_injective
    settings = {'configuration': np.zeros(41, dtype=int), 'site': 20, 'steps': 10}
    differences = _automaton('difference', _elementary(30, 41), **settings).differences
    assert differences.sum(axis=1).tolist() == [1, 3, 3, 6, 4, 9, 5, 12, 7, 12, 11]
    golden = _subshift_system(LatticeSpace(10), GOLDEN_PAIR, ((1, 1),))
    windows = [LatticeSpace(length, boundary='fixed') for length in (10, 20)]
    entropies = _automaton('entropy', golden, windows=windows).entropy
    assert entropies.tolist() == pytest.approx([0.7169925001442313, 0.7056179025755023])
    natural = _automaton('entropy', golden, windows=windows[1:], base=np.e).entropy
    assert natural.tolist() == pytest.approx([0.4890970597228316])
    majority = _automaton_system(LatticeSpace(12), BoxNeighborhood(), TotalisticSpec({2: 1, 3: 1}))
    image = _automaton('entropy', majority, windows=[LatticeSpace(12, boundary='fixed')]).entropy
    assert image.tolist() == pytest.approx([0.8954196310386876])


def test_rule_90_matrix_and_step_agree():
    """Test that rule 90 on a ring of 6 cells steps as its matrix over GF(2) maps, with recorded entries and symbol."""
    rule90 = _automaton_system(LatticeSpace(6), LatticeNeighborhood(((-1,), (1,))), LinearRuleSpec((1, 1), 'GF(2)'))
    configuration = np.array([1, 0, 1, 1, 0, 0])
    found = _automaton('matrix', rule90).matrix
    entries = np.asarray([[int(entry) for entry in row] for row in found.rows()])
    np.testing.assert_array_equal(_step(rule90, configuration), entries @ configuration % 2)
    assert entries.tolist() == [
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ]
    assert found.rank() == 4
    assert str(_automaton('polynomial', rule90).polynomial) == 'x0^-1 + x0'


def test_reference_values_of_polynomials_and_matrices():
    """Test that the Laplacian over QQ and GF(3) and a rule on a plane keep their recorded matrices and symbols."""
    laplacian = _automaton_system(LatticeSpace(5), BoxNeighborhood(), LinearRuleSpec((1, -2, 1), 'QQ'))
    polynomial = _automaton('polynomial', laplacian).polynomial
    assert str(polynomial) == 'x0^-1 - 2 + x0'
    assert sorted(polynomial.exponents()) == [-1, 0, 1]
    assert polynomial(1) == 0
    assert _automaton('matrix', laplacian).matrix.rank() == 4
    modular = _automaton_system(LatticeSpace(5), BoxNeighborhood(), LinearRuleSpec((1, -2, 1), 'GF(3)'))
    assert str(_automaton('polynomial', modular).polynomial) == 'x0^-1 + 1 + x0'
    assert [int(entry) for entry in _automaton('matrix', modular).matrix.rows()[0]] == [1, 1, 0, 0, 1]
    offsets = LatticeNeighborhood(((1, 0), (0, -1)))
    plane = _automaton_system(LatticeSpace((3, 3)), offsets, LinearRuleSpec((1, 1), 'GF(2)'))
    assert str(_automaton('polynomial', plane).polynomial) == 'x0 + x1^-1'


def test_matrix_on_a_ball_and_polynomial_without_offsets_raise():
    """Test that a matrix on a lattice with fixed edges and a polynomial on a memory set are rejected."""
    fixed = _automaton_system(LatticeSpace(5, boundary='fixed'), BoxNeighborhood(), LinearRuleSpec((1, 1, 1)))
    with pytest.raises(AnalysisError, match='not on a ball'):
        _automaton('matrix', fixed)
    group = _automaton_system(_cycle(5), MemoryNeighborhood(('a',)), LinearRuleSpec((1,)))
    with pytest.raises(AnalysisError, match='with offsets'):
        _automaton('polynomial', group)


def test_local_map_is_read_by_the_decisions():
    """Test that Life keeps the hash of its recorded local map, read from the run on a torus of 3 x 3 cells."""
    neighbors = MooreNeighborhood().prepare().offsets_
    positions = [4, *[(1 + first) * 3 + (1 + second) for first, second in neighbors]]
    configurations = np.zeros((512, 9), dtype=int)
    configurations[:, positions] = list(itertools.product((0, 1), repeat=9))
    life = _automaton_system(LatticeSpace((3, 3)), MooreNeighborhood(), LIFE)
    local_map = _step(life, configurations)[:, 4]
    assert local_map.size == 512
    assert int(local_map.sum()) == 140
    assert hashlib.sha256(local_map.astype(np.int64).tobytes()).hexdigest()[:16] == '9c774357e78bfee0'


def test_rule_values_outside_the_states_raise():
    """Test that a rule returning a value outside 0 to states - 1 is rejected."""
    wrong = _automaton_system(LatticeSpace(5), BoxNeighborhood(), TotalisticSpec(lambda total: total))
    with pytest.raises(AnalysisError, match='outside the integers 0 to 1'):
        automaton(wrong)


def test_rules_that_are_no_map_of_configurations_raise():
    """Test that a stochastic rule, an asynchronous rule, a rule naming a parameter, and two rules are rejected."""
    rule = TotalisticSpec({1: 1})
    space = LatticeSpace(5)
    with pytest.raises(AnalysisError, match='synchronous'):
        automaton(_automaton_system(space, BoxNeighborhood(), rule, schedule='random-sequential'))

    def needs_degree(*values, degree):
        """Return a rule value that needs the degree of the site."""
        return sum(values) // degree

    with pytest.raises(AnalysisError, match='of the cells alone'):
        automaton(_automaton_system(space, BoxNeighborhood(), needs_degree))
    double = _automaton_system(space, BoxNeighborhood(), rule)
    double.interactions = [*double.interactions, *double.interactions]
    with pytest.raises(AnalysisError, match='one local rule'):
        automaton(double)


def test_composition_runs_in_a_simulation():
    """Test that the system of a composition runs as the identity in a simulation."""
    shift = _automaton_system(LatticeSpace(8), LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
    back = _automaton('inverse', shift).inverse
    composition = _automaton('compose', shift, second=back).composition
    start = np.array([1, 0, 1, 1, 0, 0, 0, 1])
    cells = LatticeElement('cells', element=CellElement, space=shift.elements[0].space, values={'alive': start})
    system = System(elements=[cells], interactions=composition.interactions)
    trajectory = simulate(system, until=1, dt=1)
    np.testing.assert_array_equal(trajectory.values('cells.alive')[-1].astype(int), start)


def test_golden_mean_shift_counts_follow_fibonacci():
    """Test that the words of length n of the golden mean shift number F(n + 2), with the recorded larger counts."""
    counts = [
        _automaton('count', _subshift_system(LatticeSpace(length, boundary='fixed'), GOLDEN_PAIR, ((1, 1),))).count
        for length in range(1, 11)
    ]
    assert counts == [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    ring = _subshift_system(LatticeSpace(6), GOLDEN_PAIR, ((1, 1),))
    assert _automaton('count', ring).count == 18
    long_window = _subshift_system(LatticeSpace(30, boundary='fixed'), GOLDEN_PAIR, ((1, 1),))
    assert _automaton('count', long_window).count == 2178309


def test_violations_mark_the_forbidden_patterns():
    """Test that two adjacent ones are flagged at the first of them, and that the edge of a window is not checked."""
    window = _subshift_system(LatticeSpace(5, boundary='fixed'), GOLDEN_PAIR, ((1, 1),))
    violations = _automaton('violations', window, configuration=[0, 1, 1, 0, 1]).violations
    np.testing.assert_array_equal(violations, [False, True, False, False, False])
    assert _automaton('admissible', window, configuration=[1, 0, 1, 0, 1]).admissible
    ring = _subshift_system(LatticeSpace(5), GOLDEN_PAIR, ((1, 1),))
    assert _automaton('violations', ring, configuration=[1, 0, 0, 0, 1]).violations.tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert not _automaton('admissible', ring, configuration=[1, 0, 1, 0, 1]).admissible


def test_reference_values_of_samples():
    """Test that the golden mean shift keeps its recorded samples with fixed seeds, from zeros and from a start."""
    ring = _subshift_system(LatticeSpace(12), GOLDEN_PAIR, ((1, 1),))
    assert _automaton('sample', ring, sweeps=10, random_state=0).sample.tolist() == [
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
    ]
    window = _subshift_system(LatticeSpace(12, boundary='fixed'), GOLDEN_PAIR, ((1, 1),))
    assert np.flatnonzero(_automaton('sample', window, sweeps=5, random_state=7).sample).tolist() == [0, 8, 10]
    initial = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    small = _subshift_system(LatticeSpace(10), GOLDEN_PAIR, ((1, 1),))
    sample = _automaton('sample', small, sweeps=3, configuration=initial, random_state=3).sample
    assert sample.tolist() == [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]


def test_reference_values_on_a_group_and_with_three_values():
    """Test that subshifts on a cyclic group and over three values keep their recorded counts and samples."""
    golden = _subshift_system(_cycle(12), MemoryNeighborhood(('1', 'a')), ((1, 1),))
    assert _automaton('count', golden).count == 322
    sample = _automaton('sample', golden, sweeps=20, random_state=0).sample
    assert sample.tolist() == [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    violations = _automaton('violations', golden, configuration=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).violations
    assert np.flatnonzero(violations).tolist() == [0]
    window = _subshift_system(LatticeSpace(5, boundary='fixed'), GOLDEN_PAIR, ((0, 2), (2, 0)), 3)
    assert _automaton('count', window).count == 99
    ring = _subshift_system(LatticeSpace(8), GOLDEN_PAIR, ((0, 2), (2, 0)), 3)
    assert _automaton('sample', ring, sweeps=5, random_state=1).sample.tolist() == [0, 1, 1, 1, 1, 0, 0, 1]


def test_golden_mean_shift_samples_admissible_configurations():
    """Test that the Markov chain keeps configurations without two adjacent ones, and rejects a bad start."""
    golden = _subshift_system(_cycle(60), MemoryNeighborhood(('1', 'a')), ((1, 1),))
    sample = _automaton('sample', golden, sweeps=20, random_state=0).sample
    assert _automaton('admissible', golden, configuration=sample).admissible
    assert 0 < sample.mean() < 0.5
    with pytest.raises(AnalysisError, match='forbidden pattern'):
        _automaton('sample', golden, configuration=np.ones(60))
