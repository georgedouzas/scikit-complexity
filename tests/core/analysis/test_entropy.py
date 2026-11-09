"""Test the entropy analysis of distributions, blocks, ordinal patterns, and regularity."""

import math
import time
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from skcomplex.core.analysis import entropy, markov
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
from skcomplex.systems.physics.nonlinear_oscillators import VanDerPolSystem
from skcomplex.systems.physics.statistical_mechanics import IsingSystem

RUNS = 20_000


def _entropy(kind: str, source: object, **settings: object) -> np.ndarray:
    """Return the entropies of a source for one kind."""
    return entropy(source, kind=kind, **settings).entropy


def _logistic(r: float, size: int, start: float = 0.3) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = start
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def _pink_noise(rng: np.random.Generator, size: int) -> np.ndarray:
    """Return 1/f noise shaped in Fourier space."""
    frequencies = np.fft.rfftfreq(size)
    amplitudes = np.zeros_like(frequencies)
    amplitudes[1:] = 1 / np.sqrt(frequencies[1:])
    return np.fft.irfft(amplitudes * np.exp(2j * np.pi * rng.random(frequencies.size)), n=size)


def _previous_draws() -> tuple[list[np.ndarray], np.ndarray]:
    """Return the fixed sequences and the noise the previous in-house estimators were evaluated on."""
    rng = np.random.default_rng(7)
    sequences = [rng.integers(0, 2, 500), rng.integers(0, 4, 300), rng.integers(0, 3, 1000)]
    return sequences, rng.standard_normal(3000)


def test_shannon_entropy_of_probabilities():
    """Test the Shannon entropy of a probability vector and of symbols."""
    assert _entropy('shannon', [0.5, 0.25, 0.25], probabilities=True)[0] == pytest.approx(1.5)
    assert _entropy('shannon', [0, 1, 0, 1])[0] == pytest.approx(1.0)


def test_renyi_entropy_of_a_uniform_distribution():
    """Test that every order of the Rényi entropy of a uniform law on 8 outcomes gives 3 bits."""
    uniform = np.full(8, 1 / 8)
    for order in (0, 0.5, 1, 2, math.inf):
        assert _entropy('renyi', uniform, order=order, probabilities=True)[0] == pytest.approx(3, abs=1e-12)


def test_tsallis_entropy_limit():
    """Test that the Tsallis entropy near index 1 equals the Shannon entropy in nats."""
    _, counts = np.unique(np.random.default_rng(0).poisson(3.0, 10_000), return_counts=True)
    probabilities = counts / counts.sum()
    expected = _entropy('shannon', probabilities, base=math.e, probabilities=True)[0]
    near_one = _entropy('tsallis', probabilities, order=1 + 1e-9, probabilities=True)[0]
    assert near_one == pytest.approx(expected, abs=1e-6)
    assert _entropy('tsallis', [0.5, 0.5], order=1, probabilities=True)[0] == pytest.approx(math.log(2))


def test_periodic_sequence():
    """Test that the repeated sequence 0011 has entropy rate 0 and excess entropy 2 bits."""
    sequence = [0, 0, 1, 1] * 500
    assert _entropy('rate', sequence, dimension=6)[0] == pytest.approx(0.0, abs=1e-6)
    assert _entropy('excess', sequence, dimension=6)[0] == pytest.approx(2.0, abs=1e-6)
    assert _entropy('block', sequence, dimension=2)[0].tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_random_sequence_has_full_entropy_rate():
    """Test that fair random bits have an entropy rate near 1 bit and no excess entropy."""
    bits = np.random.default_rng(0).integers(0, 2, 100_000)
    assert _entropy('rate', bits, dimension=6)[0] == pytest.approx(1.0, abs=0.01)
    assert _entropy('excess', bits, dimension=6)[0] == pytest.approx(0.0, abs=0.05)


def test_permutation_entropy_of_monotone_and_random_series():
    """Test that a ramp has no permutation entropy and white noise has full permutation entropy."""
    assert _entropy('permutation', np.arange(50.0))[0] == 0.0
    noise = np.random.default_rng(4).normal(size=50_000)
    assert _entropy('permutation', noise)[0] == pytest.approx(1.0, abs=0.01)


def test_permutation_entropy_matches_previous_values():
    """Test that the permutation entropy equals the previous in-house estimate on a fixed seed."""
    _, values = _previous_draws()
    assert _entropy('permutation', values, dimension=4, normalize=False)[0] == pytest.approx(
        4.5814617588460695,
        abs=1e-8,
    )
    assert _entropy('permutation', values, dimension=4, delay=2)[0] == pytest.approx(1, abs=0.01)


def test_permutation_entropy_of_fields_of_two_domains():
    """Test that a hot Ising lattice gives above 0.98 and a Gray-Scott pattern at least 0.1 less."""
    settings = {'dimension': (2, 2), 'ties': 'random', 'random_state': 0}
    spins = simulate(IsingSystem(size=64, temperature=10.0), until=20, random_state=0)
    hot = _entropy('permutation', spins, **settings)[-1]
    spots = simulate(GrayScottSystem(), random_state=0)
    pattern = _entropy('permutation', spots, **settings)[-1]
    assert hot > 0.98
    assert pattern < hot - 0.1


def test_lempel_ziv_complexity():
    """Test the LZ76 phrase counts of fixed sequences and the normalised complexity of random bits."""
    kaspar = [int(bit) for bit in '0001101001000101']
    assert _entropy('lempel-ziv', kaspar, normalize=False)[0] == 6
    sequences, _ = _previous_draws()
    assert [_entropy('lempel-ziv', item, normalize=False)[0] for item in sequences] == [62.0, 75.0, 165.0]
    flips = np.random.default_rng(10).integers(0, 2, 100_000)
    assert _entropy('lempel-ziv', flips)[0] == pytest.approx(1, abs=0.1)


@pytest.mark.benchmark
def test_lempel_ziv_complexity_of_a_long_sequence_is_fast():
    """Test that the complexity of 10^5 coin flips returns within one second on an idle machine."""
    flips = np.random.default_rng(10).integers(0, 2, 100_000)
    _entropy('lempel-ziv', flips[:100])
    start = time.perf_counter()
    _entropy('lempel-ziv', flips)
    assert time.perf_counter() - start < 1


def test_sample_entropy_of_white_noise():
    """Test that 10,000 values of white noise give a sample entropy within 0.1 of 2.2 nats."""
    noise = np.random.default_rng(2).standard_normal(10_000)
    assert _entropy('sample', noise, dimension=2)[0] == pytest.approx(2.2, abs=0.1)
    assert _entropy('approximate', noise, dimension=2)[0] > 1.5


def test_sample_entropy_of_periodic_orbits_on_two_domains():
    """Test that the Van der Pol limit cycle and the logistic map at r = 3.2 give sample entropies below 0.05."""
    trajectory = simulate(VanDerPolSystem().prepare().system_, until=100.0, dt=0.01)
    cycle = trajectory.values('oscillator.x')[len(trajectory.times) // 2 :]
    assert _entropy('sample', cycle, dimension=2)[0] < 0.05
    assert _entropy('sample', _logistic(3.2, 3000), dimension=2)[0] < 0.05


def test_multiscale_entropy_signature():
    """Test that white noise loses entropy with the scale while 1/f noise keeps it within 15 percent."""
    rng = np.random.default_rng(3)
    white = _entropy('multiscale', rng.standard_normal(20_000), dimension=2)[0]
    pink = _entropy('multiscale', _pink_noise(rng, 20_000), dimension=2)[0]
    assert np.all(np.diff(white) < 0)
    assert np.all(np.abs(pink / pink[0] - 1) < 0.15)


def test_spectral_entropy_of_a_tone_and_of_noise():
    """Test that a pure sine gives below 0.1 and white noise above 0.9."""
    tone = np.sin(2 * np.pi * 64 * np.arange(4096) / 4096)
    assert _entropy('spectral', tone)[0] < 0.1
    assert _entropy('spectral', np.random.default_rng(4).standard_normal(4096))[0] > 0.9


def test_every_series_of_a_trajectory_gets_an_entropy():
    """Test that the named series of a trajectory each get an entropy, labelled by name."""
    trajectory = simulate(LotkaVolterraSystem(), until=5.0, random_state=0)
    analysis = entropy(trajectory, kind='sample', variables=['prey.x', 'predators.x'], dimension=2)
    assert analysis.labels == ['prey.x', 'predators.x']
    assert analysis.entropy.shape == (2,)


def test_unknown_kind_lists_the_kinds():
    """Test that an unknown kind raises an error that lists the kinds."""
    with pytest.raises(AnalysisError, match='shannon, renyi'):
        entropy([0, 1, 0, 1], kind='unknown')


def test_setting_the_kind_ignores_is_rejected():
    """Test that a changed setting the kind does not read raises an error naming it."""
    with pytest.raises(AnalysisError, match='does not read tolerance'):
        entropy([0, 1, 0, 1], kind='shannon', tolerance=0.5)


def test_invalid_settings_are_rejected():
    """Test that unknown ties, a negative dimension, and a window for another kind raise errors."""
    with pytest.raises(AnalysisError, match='ties'):
        entropy([0, 1, 0, 1], kind='permutation', ties='first')
    with pytest.raises(AnalysisError, match='dimension=-1'):
        entropy([0, 1, 0, 1], kind='sample', dimension=-1)
    with pytest.raises(AnalysisError, match='window'):
        entropy([0, 1, 0, 1], kind='sample', dimension=(2, 2))


def test_result_is_frozen():
    """Test that the result of an entropy is a frozen value."""
    result = entropy([0, 1, 1], kind='renyi', order=3.0)
    with pytest.raises(FrozenInstanceError):
        result.entropy = np.zeros(1)


def _ehrenfest(rng: np.random.Generator, balls: int, steps: int) -> np.ndarray:
    """Return an ensemble of lazy Ehrenfest urns of `balls` balls started all in the first urn, runs by steps."""
    states = np.full(RUNS, balls)
    frames = [states.copy()]
    for _ in range(steps):
        active = rng.random(RUNS) < 0.5
        moved = rng.random(RUNS) < states / balls
        states = states + np.where(active, np.where(moved, -1, 1), 0)
        frames.append(states.copy())
    return np.asarray(frames).T


def _birth_death(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an ensemble of a birth-death chain on 0 to 20 started at 20, and its transition matrix."""
    size = 21
    matrix = np.zeros((size, size))
    for state in range(size):
        up, down = (0.3 if state < size - 1 else 0.0), (0.4 if state > 0 else 0.0)
        if state < size - 1:
            matrix[state, state + 1] = up
        if state > 0:
            matrix[state, state - 1] = down
        matrix[state, state] = 1 - up - down
    states = np.full(RUNS, size - 1)
    frames = [states.copy()]
    cumulative = matrix.cumsum(axis=1)
    for _ in range(steps):
        states = (rng.random(RUNS)[:, None] > cumulative[states]).sum(axis=1)
        frames.append(states.copy())
    return np.asarray(frames).T, matrix


def test_ensemble_entropy_h_theorem_of_a_relaxing_ensemble():
    """Test that the divergence to the stationary law of two relaxing chains falls and ends below 0.01 bits."""
    rng = np.random.default_rng(1)
    balls = 20
    urn = _ehrenfest(rng, balls, 150)
    binomial = np.asarray([math.comb(balls, count) for count in range(balls + 1)]) / 2**balls
    chain, matrix = _birth_death(rng, 600)
    stationary = markov(matrix, kind='stationary', matrix='transition').stationary
    for samples, law in ((urn, binomial), (chain, stationary)):
        relaxation = entropy(samples, kind='ensemble', reference=law)
        noise = 5 * relaxation.error
        assert relaxation.entropy.shape == relaxation.divergence.shape == (samples.shape[1],)
        assert np.all(np.diff(relaxation.divergence) <= noise)
        assert relaxation.divergence[-1] < 0.01


def test_ensemble_entropy_checks_the_reference_law():
    """Test that the ensemble kind needs a reference law and rejects outcomes outside it."""
    with pytest.raises(AnalysisError, match='needs the reference law'):
        entropy([0, 1, 0, 1], kind='ensemble')
    with pytest.raises(AnalysisError, match='Outcomes must lie'):
        entropy([[0, 3]], kind='ensemble', reference=[0.5, 0.5])
