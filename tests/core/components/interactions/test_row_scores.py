"""Test the row scores and the infinite intensity of `_sampling` in `skcomplex.core.components.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RowScoreSpec, StateSpec
from skcomplex.systems.physics.common.spins import ThermalSamplingInteraction

INFINITE = float('inf')
LETTERS = 'ACGU'


class BitElement(Element):
    """Binary site with a weight."""

    def __init__(
        self,
        label,
        s: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1,
        weight: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.s = s
        self.weight = weight


class SpinElement(Element):
    """Spin site."""

    def __init__(self, label, s: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.s = s


class LetterElement(Element):
    """Site of a sequence of letters."""

    def __init__(self, label, letter: Annotated[str, StateSpec(domain=FiniteEnumeratedSet(list(LETTERS)))] = 'A'):
        super().__init__(label)
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
    bits = LatticeElement('bits', element=BitElement, space=LatticeSpace(shape))
    sampling = SamplingInteraction(
        'walk',
        target='bits',
        state='s',
        objective=objective,
        move=settings.pop('move', _flip_site),
        intensity=intensity,
        **settings,
    )
    return System(elements=[bits], interactions=[sampling], time=DiscreteTime())


def test_row_score_never_grows_and_reaches_zero_at_infinite_intensity():
    """Test that the number of ones never grows and reaches 0 before time 2000 in more than 99 of 100 seeds."""
    reached = 0
    for seed in range(100):
        trajectory = simulate(_bits(), until=500, random_state=seed)
        scores = trajectory.values('walk.score')[:, 0]
        assert np.all(np.diff(scores) <= 0)
        reached += scores[-1] == 0
    assert reached > 99


def test_row_score_at_unit_intensity_matches_the_boltzmann_mean():
    """Test that the mean number of ones at intensity 1 matches `20 / (1 + e)` within 3%."""
    trajectory = simulate(_bits(intensity=1.0), until=20000, random_state=0)
    assert trajectory.values('walk.score')[2000:].mean() == pytest.approx(20 / (1 + np.e), rel=0.03)


def test_row_score_glauber_rule_matches_the_boltzmann_mean():
    """Test that the Glauber rule samples the same mean number of ones."""
    system = _bits(intensity=1.0, rule='glauber')
    trajectory = simulate(system, until=20000, random_state=0)
    assert trajectory.values('walk.score')[2000:].mean() == pytest.approx(20 / (1 + np.e), rel=0.03)


def test_row_score_keeps_a_neutral_walk_on_its_target():
    """Test that a score of 0 on a phenotype and 1 off it keeps every recorded row on the phenotype at intensity oo."""

    def score(row):
        return float(row[:3].sum() != 3)

    trajectory = simulate(_bits((4, 10), RowScoreSpec(score), neighborhood=None), until=200, random_state=0)
    states = trajectory.lattice('bits.s').astype(float)
    assert np.all(states[:, :, :3] == 1)
    assert np.any(states[-1] != 1)
    assert np.all(trajectory.values('walk.score') == 0)


def test_row_score_gives_every_row_its_own_draws():
    """Test that independent rows take different walks."""
    trajectory = simulate(_bits((3, 20)), until=5, random_state=0)
    states = trajectory.lattice('bits.s').astype(float)[-1]
    assert not np.array_equal(states[0], states[1])


def test_row_score_proposes_its_declared_number_of_moves():
    """Test that a row score takes its number of proposals per unit of time."""
    system = _bits((1, 20), RowScoreSpec(np.sum, proposals=5), intensity=1e-9)
    trajectory = simulate(system, until=1, random_state=0)
    changed = np.abs(np.diff(trajectory.lattice('bits.s').astype(float), axis=0)).sum()
    assert 1 <= changed <= 5
    assert changed % 2 == 5 % 2


def test_row_score_reads_the_parameters_it_names():
    """Test that a row score receives the parameters of the system it names, one row of a parameter per site."""
    seen = []

    def score(row, weight):
        seen.append(np.shape(weight))
        return float(np.sum(row * weight))

    bits = LatticeElement(
        'bits',
        element=BitElement,
        space=LatticeSpace((2, 4)),
        values={'weight': np.ones((2, 4))},
    )
    sampling = SamplingInteraction('walk', target='bits', state='s', objective=RowScoreSpec(score), move=_flip_site)
    system = System(elements=[bits], interactions=[sampling], time=DiscreteTime())
    simulate(system, until=1, random_state=0)
    assert set(seen) == {(4,)}


@pytest.mark.parametrize(
    ('move', 'settings'),
    [('flip', {}), ('random-value', {'values': (-1.0, 1.0)}), ('rotate', {'width': 0.5}), ('gaussian', {'width': 0.5})],
)
def test_row_score_takes_named_moves_at_one_site(move, settings):
    """Test that a named move changes at most one site of a row per proposal."""
    spins = LatticeElement('spins', element=SpinElement, space=LatticeSpace((1, 8)))
    sampling = SamplingInteraction(
        'walk',
        target='spins',
        state='s',
        objective=RowScoreSpec(lambda row: 0.0),
        move=move,
        **settings,
    )
    system = System(elements=[spins], interactions=[sampling], time=DiscreteTime())
    trajectory = simulate(system, until=4, random_state=0)
    assert np.all(np.sum(np.diff(trajectory.lattice('spins.s'), axis=0) != 0, axis=2) <= 1)


def test_row_score_cools_with_time():
    """Test that a cooling that grows with time heats a row score, which then accepts raising moves."""
    system = _bits(intensity=1e6, cooling=lambda t: 1e9 * (1 + t))
    trajectory = simulate(system, until=50, random_state=0)
    assert np.any(np.diff(trajectory.values('walk.score')[:, 0]) > 0)


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
    letters = LatticeElement('letters', element=LetterElement, space=LatticeSpace((2, 6)))
    sampling = SamplingInteraction(
        'walk',
        target='letters',
        state='letter',
        objective=RowScoreSpec(lambda row: 0.0),
        move=_mutate,
    )
    system = System(elements=[letters], interactions=[sampling], time=DiscreteTime())
    trajectory = simulate(system, until=5, random_state=0)
    assert set(trajectory.values('letters.letter').ravel()) <= set(LETTERS)


def test_row_score_rejects_a_function_that_is_not_callable():
    """Test that a row score of a value raises an error, which the compiler reports as a structure issue."""
    with pytest.raises(ValidationError, match='callable'):
        _bits(objective=RowScoreSpec(1.0)).interactions[0].prepare()
    report = PassCompiler().prepare(_bits(objective=RowScoreSpec(1.0))).report_
    assert not report.ok
    assert 'structure' in {issue.code for issue in report.issues}


def test_row_score_rejects_too_few_proposals():
    """Test that a row score proposes at least one move."""
    with pytest.raises(ValidationError, match='at least one'):
        RowScoreSpec(np.sum, proposals=0).prepare()


@pytest.mark.parametrize(('settings'), [{'rule': 'heat-bath'}, {'move': 'cluster'}])
def test_row_score_rejects_a_heat_bath_without_values_and_clusters(settings):
    """Test that a row score takes no heat-bath rule without its values, and no cluster move."""
    with pytest.raises(ValidationError, match='row score'):
        _bits(**settings).interactions[0].prepare()


def test_row_score_heat_bath_matches_the_boltzmann_mean():
    """Test that the heat-bath rule on a row samples the mean number of ones at intensity 1."""
    system = _bits(intensity=1.0, rule='heat-bath', move='random-value', values=(0.0, 1.0))
    trajectory = simulate(system, until=20000, random_state=0)
    assert trajectory.values('walk.score')[2000:].mean() == pytest.approx(20 / (1 + np.e), rel=0.03)


def test_row_score_heat_bath_at_infinite_intensity_never_raises_the_score():
    """Test that the heat-bath rule on a row at infinite intensity keeps only values of the lowest score."""
    system = _bits(rule='heat-bath', move='random-value', values=(0.0, 1.0))
    trajectory = simulate(system, until=300, random_state=0)
    scores = trajectory.values('walk.score')[:, 0]
    assert np.all(np.diff(scores) <= 0)
    assert scores[-1] == 0


def test_row_score_reports_a_target_off_a_lattice():
    """Test that the compiler reports a row score of a group on a network as a law-space issue."""
    space = NetworkSpace(nx.path_graph(4))
    spins = LatticeElement('spins', element=SpinElement, space=space)
    sampling = SamplingInteraction('walk', target='spins', state='s', objective=RowScoreSpec(np.sum), move=_flip_site)
    system = System(elements=[spins], interactions=[sampling], time=DiscreteTime())
    report = PassCompiler().prepare(system).report_
    assert 'law-space' in {issue.code for issue in report.issues}


def _ising(namespace):
    """Return the local Ising energy."""
    return -namespace.target.s * neighbors(namespace.target.other.s)


def _total_energy(spins):
    """Return the Ising energy of a periodic lattice."""
    return -(spins * np.roll(spins, 1, axis=-1)).sum(axis=(-1, -2)) - (spins * np.roll(spins, 1, axis=-2)).sum(
        axis=(-1, -2),
    )


def _spins_at(sampling):
    """Return the lattice frames of random spins under a sampling."""
    values = {'s': np.random.default_rng(0).choice([-1.0, 1.0], size=(16, 16))}
    spins = LatticeElement('spins', element=SpinElement, space=LatticeSpace((16, 16)), values=values)
    system = System(elements=[spins], interactions=[sampling], time=DiscreteTime())
    return simulate(system, until=60, random_state=0).lattice('spins.s')


@pytest.mark.parametrize('rule', ['metropolis', 'glauber', 'heat-bath'])
def test_infinite_intensity_never_raises_the_energy(rule):
    """Test that the total energy of an Ising lattice never grows from one frame to the next at infinite intensity."""
    sampling = SamplingInteraction('ising', target='spins', state='s', objective=_ising, intensity=INFINITE, rule=rule)
    energies = _total_energy(_spins_at(sampling))
    assert np.all(np.diff(energies) <= 0)
    assert energies[-1] < energies[0]


def test_infinite_intensity_accepts_moves_that_keep_the_energy():
    """Test that the Metropolis rule at infinite intensity accepts a move that keeps the objective."""
    sampling = SamplingInteraction(
        'free',
        target='spins',
        state='s',
        objective=lambda n: 0 * n.target.s,
        intensity=INFINITE,
    )
    frames = _spins_at(sampling)
    assert np.array_equal(frames[1], -frames[0])


def test_thermal_sampling_at_zero_temperature_never_raises_the_energy():
    """Test that the thin thermal subclass at temperature 0 is the infinite intensity of the general class."""
    sampling = ThermalSamplingInteraction('ising', target='spins', state='s', energy=_ising, temperature=0.0)
    energies = _total_energy(_spins_at(sampling))
    assert np.all(np.diff(energies) <= 0)
    assert energies[-1] < energies[0]


def test_thermal_sampling_takes_a_row_score_at_zero_temperature():
    """Test that the thin thermal subclass runs a row score at temperature 0."""
    bits = LatticeElement('bits', element=BitElement, space=LatticeSpace((1, 20)))
    sampling = ThermalSamplingInteraction(
        'walk',
        target='bits',
        state='s',
        energy=RowScoreSpec(np.sum),
        move=_flip_site,
        temperature=0.0,
    )
    system = System(elements=[bits], interactions=[sampling], time=DiscreteTime())
    trajectory = simulate(system, until=100, random_state=0)
    assert np.all(np.diff(trajectory.values('walk.score')[:, 0]) <= 0)


def test_intensity_rejects_zero_and_negative_values():
    """Test that the domain of the intensity holds infinity but neither zero nor negative values."""
    for intensity in (0.0, -1.0):
        report = PassCompiler().prepare(_bits(intensity=intensity)).report_
        assert not report.ok
