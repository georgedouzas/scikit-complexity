"""Test avalanche statistics, crackling-noise scaling, branching ratios, and transitions of order parameters."""

import math

import numpy as np
import pytest
from scipy import optimize

from skcomplex.core.analysis import (
    Avalanches,
    avalanches,
    branching_ratio,
    crackling_noise,
    fields_of,
    fit_avalanches,
    fit_order_exponent,
    order_parameter_sweep,
    replica_overlap,
)
from skcomplex.core.base import AnalysisError
from skcomplex.core.runtime import Simulation
from skcomplex.models import create

from .conftest import ISING_TEMPERATURES

CRITICAL = 2 / math.log(1 + math.sqrt(2))
DURATIONS = (20, 40, 80)
MAX_GENERATIONS = 5000


def _galton_watson(rng: np.random.Generator, trees: int) -> Avalanches:
    """Return the avalanches of critical Galton-Watson trees with Poisson offspring of mean 1, one generation a step."""
    alive, active = np.arange(trees), np.ones(trees, dtype=np.int64)
    sizes, durations = np.zeros(trees), np.zeros(trees)
    history: list[tuple[np.ndarray, np.ndarray]] = []
    while alive.size and len(history) < MAX_GENERATIONS:
        history.append((alive, active))
        sizes[alive] += active
        durations[alive] += 1
        active = rng.poisson(active)
        alive, active = alive[active > 0], active[active > 0]
    shapes = []
    for duration in DURATIONS:
        chosen = np.flatnonzero(durations == duration)
        profiles = np.asarray([members[np.searchsorted(ids, chosen)] for ids, members in history[:duration]]).T
        shapes.extend(profiles.astype(float))
    return Avalanches(sizes, durations, shapes)


def _branching_activity(rng: np.random.Generator, ratio: float, drive: float, steps: int) -> np.ndarray:
    """Return the activity of a driven branching process, each active unit triggering Poisson descendants."""
    activity = np.zeros(steps)
    activity[0] = drive / (1 - ratio)
    for t in range(1, steps):
        activity[t] = rng.poisson(ratio * activity[t - 1] + drive)
    return activity


def _kuramoto_order(rng: np.random.Generator, coupling: float, oscillators: int) -> np.ndarray:
    """Return the order parameter of Kuramoto oscillators with Lorentzian frequencies of width 1 after relaxation."""
    frequencies = np.tan(np.pi * (rng.random(oscillators) - 0.5))
    phases = rng.uniform(-np.pi, np.pi, oscillators)
    orders = []
    for step in range(6000):
        order = np.mean(np.exp(1j * phases))
        phases = phases + 0.05 * (frequencies + coupling * np.abs(order) * np.sin(np.angle(order) - phases))
        if step >= 2000:
            orders.append(np.abs(order))
    return np.asarray(orders)


def test_avalanches_same_statistics_on_a_sandpile_and_an_earthquake_model():
    """Test the size exponents of a Bak-Tang-Wiesenfeld sandpile and an Olami-Feder-Christensen model."""
    sandpile = Simulation(create('sandpile', shape=(128, 128)), dt=1000).run(until=150_000).trajectory_
    earthquakes = Simulation(create('earthquakes', shape=(64, 64), alpha=0.2), dt=1000).run(until=110_000).trajectory_
    piles = avalanches([record for record in sandpile.events if record.time > 50_000 and record.size > 0])
    quakes = avalanches([record for record in earthquakes.events if record.time > 10_000 and record.size > 0])
    for found, (low, high) in ((piles, (1.15, 1.35)), (quakes, (1.7, 2.1))):
        laws = fit_avalanches(found, size_range=(10, None), duration_range=(1, None))
        assert low <= laws.size.alpha <= high
        ratio, p_value = laws.size.comparisons['exponential']
        assert ratio > 0
        assert p_value < 0.05


def test_avalanches_mean_field_branching():
    """Test that critical Galton-Watson avalanches have size exponent 3/2 and duration exponent 2."""
    found = _galton_watson(np.random.default_rng(0), 100_000)
    laws = fit_avalanches(found, size_range=(10, None), duration_range=(10, None))
    assert laws.size.alpha == pytest.approx(1.5, abs=0.05)
    assert laws.duration.alpha == pytest.approx(2, abs=0.1)


def test_crackling_noise_branching_process_scaling():
    """Test that the mean size grows as the squared duration, agrees with the laws, and the shapes collapse."""
    laws = fit_avalanches(
        _galton_watson(np.random.default_rng(1), 100_000),
        size_range=(10, None),
        duration_range=(10, None),
    )
    crackling = crackling_noise(_galton_watson(np.random.default_rng(2), 1_000_000), laws, DURATIONS, min_duration=10)
    assert crackling.exponent == pytest.approx(2, abs=0.1)
    assert crackling.exponent == pytest.approx(crackling.predicted, abs=0.15)
    assert crackling.shape_spread < 0.1
    with pytest.raises(AnalysisError, match='Fewer than 2 durations'):
        crackling_noise(Avalanches(np.ones(3), np.ones(3), []))


def test_avalanches_of_an_activity_series():
    """Test that runs of activity above a threshold are avalanches with their profiles."""
    found = avalanches(np.asarray([0, 2, 3, 0, 1, 0]), threshold=0.5)
    assert found.sizes.tolist() == [4.0, 0.5]
    assert [shape.size for shape in found.shapes] == [2, 1]


def test_branching_ratio_subsampled_network():
    """Test that a branching network with ratio 0.98 gives 0.98 from all units and from 5 percent of them."""
    rng = np.random.default_rng(2)
    activity = _branching_activity(rng, 0.98, 20.0, 100_000)
    assert branching_ratio(activity) == pytest.approx(0.98, abs=0.01)
    assert branching_ratio(rng.binomial(activity.astype(int), 0.05)) == pytest.approx(0.98, abs=0.01)


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
    assert branching_ratio(np.asarray(spikes, dtype=float)) == pytest.approx(1, abs=0.02)
    cases = _branching_activity(rng, 0.9, 5.0, 50_000)
    assert branching_ratio(cases) == pytest.approx(0.9, abs=0.03)


def test_order_parameter_sweep_same_susceptibility_peak_on_two_transitions(ising_magnetizations):
    """Test that the susceptibility peaks of a 64 by 64 Ising lattice and of Kuramoto oscillators locate T_c and K_c."""
    ising = order_parameter_sweep(
        ISING_TEMPERATURES,
        ising_magnetizations[64],
        64**2,
        temperatures=ISING_TEMPERATURES,
    )
    assert ising.peak == pytest.approx(CRITICAL, rel=0.03)
    couplings = np.linspace(1.0, 3.0, 11)
    rng = np.random.default_rng(4)
    orders = [_kuramoto_order(rng, coupling, 2000) for coupling in couplings]
    kuramoto = order_parameter_sweep(couplings, orders, 2000)
    assert kuramoto.peak == pytest.approx(2, rel=0.1)
    assert np.all(np.diff(kuramoto.order[couplings >= 2.2]) > 0)


def test_fit_order_exponent_of_two_mean_field_transitions():
    """Test that the mean-field Ising model and the Ott-Antonsen Kuramoto model give the exponent 1/2."""
    temperatures = np.linspace(0.95, 0.995, 10)
    magnetizations = [optimize.brentq(lambda m, t=t: m - math.tanh(m / t), 1e-6, 1.0) for t in temperatures]
    assert fit_order_exponent(temperatures, magnetizations, 1.0) == pytest.approx(0.5, abs=0.05)
    couplings = np.linspace(1.02, 1.2, 10)
    orders = [
        Simulation(create('ott_antonsen', r=0.5, K=coupling, Delta=0.5), dt=1.0).run(until=400.0).trajectory_
        for coupling in couplings
    ]
    names = orders[0].problem.states.names
    final = [float(np.abs(trajectory[names[0]][-1])) for trajectory in orders]
    assert fit_order_exponent(couplings, final, 1.0, above=True) == pytest.approx(0.5, abs=0.05)
    with pytest.raises(AnalysisError, match='ordered side'):
        fit_order_exponent([1.5, 2.0], [0.1, 0.2], 1.0)


def test_replica_overlap_of_ising_lattices():
    """Test that two replicas of a 32 by 32 Ising lattice overlap at temperature 1.5 and not at temperature 5."""
    overlaps = {}
    for temperature in (1.5, 5.0):
        replicas = [
            fields_of(
                Simulation(create('ising', size=32, temperature=temperature, initial='up'), dt=1, random_state=seed)
                .run(until=500)
                .trajectory_,
            )[100:]
            for seed in (0, 1)
        ]
        overlaps[temperature] = float(np.mean(np.abs(replica_overlap(*replicas))))
    assert overlaps[1.5] > 0.9
    assert overlaps[5.0] < 0.1
    assert replica_overlap([1, -1], [1, -1]) == 1.0
