"""Test the `_sequence_sampling` module of `skcomplex.systems.biology.common.sequences`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import BaseSpace, LatticeSpace, NetworkSpace
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.sequences import RowScoreSpec, SequenceSamplingInteraction

INFINITE = float('inf')
LETTERS = 'ACGU'


class BitElement(Element):
    """Binary sites with a weight each."""

    def __init__(
        self,
        label,
        *,
        s: Annotated[float, FiniteEnumeratedSet([0, 1])] = 1,
        weight: float = 1.0,
        space: BaseSpace | None = None,
    ):
        super().__init__(label, space=space)
        self.s = s
        self.weight = weight


class LetterElement(Element):
    """Sites of a sequence of letters."""

    def __init__(
        self,
        label,
        *,
        letter: Annotated[str, FiniteEnumeratedSet(list(LETTERS))] = 'A',
        space: BaseSpace | None = None,
    ):
        super().__init__(label, space=space)
        self.letter = letter


def _flip_site(row, random):
    """Return a binary row with one random site flipped."""
    site = random.integers(row.size)
    row[site] = 1 - row[site]
    return row


def _mutate(row, random):
    """Return a row of letter indices with the value at one random site replaced by one of the other three."""
    site = random.integers(row.size)
    row[site] = (row[site] + random.integers(1, 4)) % 4
    return row


def _bits(shape=(1, 20), objective=None, intensity=INFINITE, **settings):
    """Return a binary lattice sampled by the number of its ones."""
    objective = RowScoreSpec(np.sum) if objective is None else objective
    bits = BitElement('bits', space=LatticeSpace(shape))
    sampling = SequenceSamplingInteraction(
        'walk',
        target='bits',
        state='s',
        objective=objective,
        move=settings.pop('move', _flip_site),
        intensity=intensity,
        **settings,
    )
    return System(elements=[bits], interactions=[sampling])


def test_row_score_never_grows_and_reaches_zero_at_infinite_intensity():
    """Test that the number of ones never grows and reaches 0 before time 2000 in more than 99 of 100 seeds."""
    reached = 0
    for seed in range(100):
        trajectory = _bits().prepare().simulate(until=500, random_state=seed)
        scores = trajectory.get_values('walk.score')[:, 0]
        assert np.all(np.diff(scores) <= 0)
        reached += scores[-1] == 0
    assert reached > 99


def test_row_score_at_unit_intensity_matches_the_boltzmann_mean():
    """Test that the mean number of ones at intensity 1 matches `20 / (1 + e)` within 3%."""
    trajectory = _bits(intensity=1.0).prepare().simulate(until=20000, random_state=0)
    assert trajectory.get_values('walk.score')[2000:].mean() == pytest.approx(20 / (1 + np.e), rel=0.03)


def test_row_score_glauber_rule_matches_the_boltzmann_mean():
    """Test that the Glauber rule samples the same mean number of ones."""
    system = _bits(intensity=1.0, rule='glauber')
    trajectory = system.prepare().simulate(until=20000, random_state=0)
    assert trajectory.get_values('walk.score')[2000:].mean() == pytest.approx(20 / (1 + np.e), rel=0.03)


def test_row_score_keeps_a_neutral_walk_on_its_target():
    """Test that a score of 0 on a phenotype and 1 off it keeps every recorded row on the phenotype at intensity oo."""

    def score(row):
        return float(row[:3].sum() != 3)

    trajectory = _bits((4, 10), RowScoreSpec(score)).prepare().simulate(until=200, random_state=0)
    states = trajectory.get_lattice('bits.s').astype(float)
    assert np.all(states[:, :, :3] == 1)
    assert np.any(states[-1] != 1)
    assert np.all(trajectory.get_values('walk.score') == 0)


def test_row_score_gives_every_row_its_own_draws():
    """Test that independent rows take different walks."""
    trajectory = _bits((3, 20)).prepare().simulate(until=5, random_state=0)
    states = trajectory.get_lattice('bits.s').astype(float)[-1]
    assert not np.array_equal(states[0], states[1])


def test_row_score_proposes_its_declared_number_of_moves():
    """Test that a row score takes its number of proposals per unit of time."""
    system = _bits((1, 20), RowScoreSpec(np.sum, proposals=5), intensity=1e-9)
    trajectory = system.prepare().simulate(until=1, random_state=0)
    changed = np.abs(np.diff(trajectory.get_lattice('bits.s').astype(float), axis=0)).sum()
    assert 1 <= changed <= 5
    assert changed % 2 == 5 % 2


def test_row_score_reads_the_parameters_it_names():
    """Test that a row score receives the parameters of the system it names, one row of a parameter per site."""
    seen = []

    def score(row, weight):
        seen.append(np.shape(weight))
        return float(np.sum(row * weight))

    bits = BitElement('bits', weight=np.ones((2, 4)), space=LatticeSpace((2, 4)))
    sampling = SequenceSamplingInteraction(
        'walk',
        target='bits',
        state='s',
        objective=RowScoreSpec(score),
        move=_flip_site,
    )
    system = System(elements=[bits], interactions=[sampling])
    system.prepare().simulate(until=1, random_state=0)
    assert set(seen) == {(4,)}


@pytest.mark.parametrize(
    ('move', 'settings'),
    [('flip', {}), ('random-value', {'values': (-1.0, 1.0)}), ('rotate', {'width': 0.5}), ('gaussian', {'width': 0.5})],
)
def test_row_score_takes_named_moves_at_one_site(move, settings):
    """Test that a named move changes at most one site of a row per proposal."""
    spins = Element('spins', values={'s': 1.0}, space=LatticeSpace((1, 8)))
    sampling = SequenceSamplingInteraction(
        'walk',
        target='spins',
        state='s',
        objective=RowScoreSpec(lambda row: 0.0),
        move=move,
        **settings,
    )
    system = System(elements=[spins], interactions=[sampling])
    trajectory = system.prepare().simulate(until=4, random_state=0)
    assert np.all(np.sum(np.diff(trajectory.get_lattice('spins.s'), axis=0) != 0, axis=2) <= 1)


def test_row_score_cools_with_time():
    """Test that a cooling that grows with time heats a row score, which then accepts raising moves."""
    system = _bits(intensity=1e6, cooling=lambda t: 1e9 * (1 + t))
    trajectory = system.prepare().simulate(until=50, random_state=0)
    assert np.any(np.diff(trajectory.get_values('walk.score')[:, 0]) > 0)


def test_point_mutation_changes_one_site_to_the_other_letters_equally():
    """Test that a function move changes one site, drawing each other letter with frequency 1/3 within 0.01."""
    rng = np.random.default_rng(0)
    row = rng.integers(0, 4, size=30).astype(float)
    counts = np.zeros(3)
    for _ in range(30000):
        proposal = _mutate(row.copy(), rng)
        changed = np.flatnonzero(proposal != row)
        assert changed.size == 1
        counts[int((proposal[changed[0]] - row[changed[0]]) % 4) - 1] += 1
    assert counts / counts.sum() == pytest.approx(1 / 3, abs=0.01)


def test_row_score_samples_a_categorical_state_with_labels():
    """Test that a row of letters is sampled as indices and read as its labels."""
    letters = LetterElement('letters', space=LatticeSpace((2, 6)))
    sampling = SequenceSamplingInteraction(
        'walk',
        target='letters',
        state='letter',
        objective=RowScoreSpec(lambda row: 0.0),
        move=_mutate,
    )
    system = System(elements=[letters], interactions=[sampling])
    trajectory = system.prepare().simulate(until=5, random_state=0)
    assert set(trajectory.get_values('letters.letter').ravel()) <= set(LETTERS)


def test_row_score_rejects_a_function_that_is_not_callable():
    """Test that a row score of a value raises an error, which the compiler reports as a value issue."""
    with pytest.raises(ValidationError, match='callable'):
        _bits(objective=RowScoreSpec(1.0)).interactions[0].prepare()
    report = PassCompiler().prepare(_bits(objective=RowScoreSpec(1.0))).report_
    assert not report.ok
    assert {('value', 'walk')} <= {(issue.code, issue.path) for issue in report.issues}
    assert any('callable' in issue.message for issue in report.issues)


def test_row_score_rejects_too_few_proposals():
    """Test that a row score proposes at least one move."""
    with pytest.raises(ValidationError, match='at least one'):
        RowScoreSpec(np.sum, proposals=0)._check()


@pytest.mark.parametrize(('settings'), [{'rule': 'heat-bath'}, {'move': 'cluster'}])
def test_row_score_rejects_a_heat_bath_without_values_and_clusters(settings):
    """Test that a row score takes no heat-bath rule without its values, and no cluster move."""
    with pytest.raises(ValidationError, match='row score'):
        _bits(**settings).interactions[0].prepare()


def test_row_score_heat_bath_matches_the_boltzmann_mean():
    """Test that the heat-bath rule on a row samples the mean number of ones at intensity 1."""
    system = _bits(intensity=1.0, rule='heat-bath', move='random-value', values=(0.0, 1.0))
    trajectory = system.prepare().simulate(until=20000, random_state=0)
    assert trajectory.get_values('walk.score')[2000:].mean() == pytest.approx(20 / (1 + np.e), rel=0.03)


def test_row_score_heat_bath_at_infinite_intensity_never_raises_the_score():
    """Test that the heat-bath rule on a row at infinite intensity keeps only values of the lowest score."""
    system = _bits(rule='heat-bath', move='random-value', values=(0.0, 1.0))
    trajectory = system.prepare().simulate(until=300, random_state=0)
    scores = trajectory.get_values('walk.score')[:, 0]
    assert np.all(np.diff(scores) <= 0)
    assert scores[-1] == 0


def test_row_score_reports_a_target_off_a_lattice():
    """Test that the compiler reports a row score of a group on a network as a law-space issue."""
    space = NetworkSpace(nx.path_graph(4))
    spins = Element('spins', values={'s': 1.0}, space=space)
    sampling = SequenceSamplingInteraction(
        'walk',
        target='spins',
        state='s',
        objective=RowScoreSpec(np.sum),
        move=_flip_site,
    )
    system = System(elements=[spins], interactions=[sampling])
    report = PassCompiler().prepare(system).report_
    assert 'law-space' in {issue.code for issue in report.issues}


@pytest.mark.parametrize('intensity', [0.0, -1.0])
def test_intensity_rejects_zero_and_negative_values(intensity):
    """Test that the domain of the intensity holds infinity but neither zero nor negative values."""
    with pytest.raises(ValidationError, match=r'walk\.intensity'):
        _bits(intensity=intensity).prepare()
