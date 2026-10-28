"""Test avalanche statistics, crackling-noise scaling, and branching ratios."""

import numpy as np
import pytest

from skcomplex.analysis import AvalancheAnalysis, FitAnalysis
from skcomplex.core.errors import AnalysisError
from skcomplex.engine.runtime import Simulation
from skcomplex.models import create

DURATIONS = (20, 40, 80)
MAX_GENERATIONS = 5000


def _avalanches(kind: str, source: object, **settings: object) -> AvalancheAnalysis:
    """Return the run avalanche analysis of a source for one kind."""
    return AvalancheAnalysis(kind, **settings).check().prepare(source).run()


def _exponent(values: np.ndarray, minimum: float) -> FitAnalysis:
    """Return the discrete power law fitted above a fixed lower cutoff."""
    return FitAnalysis('power', minimum=minimum, discrete=True, bootstrap=0).check().prepare(values).run()


def _galton_watson(rng: np.random.Generator, trees: int) -> np.ndarray:
    """Return the activity of critical Galton-Watson trees with Poisson offspring of mean 1, one generation a step.

    The trees follow one another, each followed by a silent step, so that every tree is one avalanche.
    """
    alive, active = np.arange(trees), np.ones(trees, dtype=np.int64)
    durations = np.zeros(trees, dtype=np.int64)
    history: list[tuple[np.ndarray, np.ndarray]] = []
    while alive.size and len(history) < MAX_GENERATIONS:
        history.append((alive, active))
        durations[alive] += 1
        active = rng.poisson(active)
        alive, active = alive[active > 0], active[active > 0]
    starts = np.concatenate([[0], np.cumsum(durations + 1)[:-1]])
    activity = np.zeros(int(np.sum(durations + 1)))
    for generation, (ids, members) in enumerate(history):
        activity[starts[ids] + generation] = members
    return activity


def _branching_activity(rng: np.random.Generator, ratio: float, drive: float, steps: int) -> np.ndarray:
    """Return the activity of a driven branching process, each active unit triggering Poisson descendants."""
    activity = np.zeros(steps)
    activity[0] = drive / (1 - ratio)
    for t in range(1, steps):
        activity[t] = rng.poisson(ratio * activity[t - 1] + drive)
    return activity


@pytest.mark.slow
@pytest.mark.parametrize(
    ('name', 'parameters', 'until', 'transient', 'bounds'),
    [
        ('sandpile', {'shape': (96, 96)}, 20_000, 12_000, (1.15, 1.35)),
        ('earthquakes', {'shape': (32, 32), 'alpha': 0.2}, 30_000, 5_000, (1.7, 2.1)),
    ],
)
def test_avalanches_same_statistics_on_a_sandpile_and_an_earthquake_model(name, parameters, until, transient, bounds):
    """Test the size exponents of a Bak-Tang-Wiesenfeld sandpile and an Olami-Feder-Christensen model."""
    trajectory = Simulation(create(name, **parameters), dt=1000).run(until=until).trajectory_
    found = _avalanches(
        'sizes',
        [record for record in trajectory.events if record.time > transient and record.size > 0],
    )
    size = _exponent(found.sizes_, 10)
    low, high = bounds
    assert low <= size.exponent_ <= high
    ratio, p_value = size.comparisons_['exponential']
    assert ratio > 0
    assert p_value < 0.05


def test_avalanches_mean_field_branching():
    """Test that critical Galton-Watson avalanches have size exponent 3/2 and duration exponent 2."""
    found = _avalanches('sizes', _galton_watson(np.random.default_rng(0), 100_000))
    assert _exponent(found.sizes_, 10).exponent_ == pytest.approx(1.5, abs=0.05)
    assert _exponent(found.durations_, 10).exponent_ == pytest.approx(2, abs=0.1)


def test_crackling_noise_branching_process_scaling():
    """Test that the mean size grows as the squared duration, agrees with the laws, and the shapes collapse."""
    found = _avalanches('sizes', _galton_watson(np.random.default_rng(1), 100_000))
    crackling = _avalanches(
        'crackling',
        _galton_watson(np.random.default_rng(2), 1_000_000),
        durations=DURATIONS,
        min_duration=10,
        size_exponent=_exponent(found.sizes_, 10).exponent_,
        duration_exponent=_exponent(found.durations_, 10).exponent_,
    )
    assert crackling.exponent_ == pytest.approx(2, abs=0.1)
    assert crackling.exponent_ == pytest.approx(crackling.predicted_, abs=0.15)
    assert crackling.shape_spread_ < 0.1
    assert sorted(crackling.mean_shapes_) == list(DURATIONS)
    with pytest.raises(AnalysisError, match='Fewer than 2 durations'):
        _avalanches('crackling', [0, 1, 0, 1, 0, 1, 0])


def test_avalanches_of_an_activity_series():
    """Test that runs of activity above a threshold are avalanches with their profiles."""
    found = _avalanches('sizes', np.asarray([0, 2, 3, 0, 1, 0]), threshold=0.5)
    assert found.sizes_.tolist() == [4.0, 0.5]
    assert [shape.size for shape in found.shapes_] == [2, 1]


def test_branching_ratio_subsampled_network():
    """Test that a branching network with ratio 0.98 gives 0.98 from all units and from 5 percent of them."""
    rng = np.random.default_rng(2)
    activity = _branching_activity(rng, 0.98, 20.0, 100_000)
    assert _avalanches('branching', activity).branching_ratio_ == pytest.approx(0.98, abs=0.01)
    subsampled = rng.binomial(activity.astype(int), 0.05)
    assert _avalanches('branching', subsampled).branching_ratio_ == pytest.approx(0.98, abs=0.01)


def test_branching_ratio_neuronal_and_epidemic_activity():
    """Test a critical stochastic neural network and a subcritical epidemic with reproduction number 0.9."""
    rng = np.random.default_rng(3)
    neurons, targets = 10_000, 10
    active = 100
    spikes = []
    for _ in range(20_000):
        hits = rng.binomial(active * targets, 1 / targets)
        active = np.unique(rng.integers(0, neurons, hits)).size + rng.binomial(neurons, 1e-4)
        spikes.append(active)
    assert _avalanches('branching', np.asarray(spikes, dtype=float)).branching_ratio_ == pytest.approx(1, abs=0.02)
    cases = _branching_activity(rng, 0.9, 5.0, 50_000)
    assert _avalanches('branching', cases).branching_ratio_ == pytest.approx(0.9, abs=0.03)


def test_invalid_settings_and_sources_are_rejected():
    """Test an unknown kind, an ignored setting, and several activity series."""
    with pytest.raises(AnalysisError, match='sizes, crackling, branching'):
        AvalancheAnalysis('cascades').check()
    with pytest.raises(AnalysisError, match='does not read max_step'):
        AvalancheAnalysis('sizes', max_step=10).check()
    with pytest.raises(AnalysisError, match='one activity series'):
        _avalanches('sizes', np.ones((5, 2)))
