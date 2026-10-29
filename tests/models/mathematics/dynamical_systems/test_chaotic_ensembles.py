"""Test the `_chaotic_ensembles` module of `skcomplex.models.mathematics.dynamical_systems`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.mathematics.dynamical_systems import (
    cubic_chaotic_map,
    doubling_map,
    skew_tent_map,
)

ENSEMBLE_SIZE = 10000


BINS = 20


STEPS = 200


STANDARD_DEVIATIONS = 4


def _assert_uniform(points):
    """Assert that every bin of the histogram of points on the unit interval is within the sampling error."""
    counts, _ = np.histogram(points, bins=BINS, range=(0.0, 1.0))
    expected = points.size / BINS
    error = math.sqrt(points.size * (1 / BINS) * (1 - 1 / BINS))
    assert np.all(np.abs(counts - expected) < STANDARD_DEVIATIONS * error)


@pytest.mark.parametrize('a', [0.3, 0.8])
def test_skew_tent_map_invariant_density_is_uniform(a):
    """Test that 10000 points iterated 200 steps fill every one of 20 bins within four binomial deviations."""
    system = skew_tent_map(ENSEMBLE_SIZE, a=a, random_state=1)
    points = Simulation(system, dt=1).check().prepare().run(until=STEPS).trajectory_.values('points.x')
    assert np.ptp(points[0]) <= 0.01
    _assert_uniform(points[-1])


def test_doubling_map_iterates_fractions_exactly():
    """Test that the doubling map follows `k' = 2k mod q` exactly well beyond the floating-point collapse."""
    system = doubling_map(ENSEMBLE_SIZE, random_state=2)
    denominator = int(system.elements[0].values['q'])
    points = Simulation(system, dt=1).check().prepare().run(until=STEPS).trajectory_.values('points.x')
    numerators = [round(value * denominator) for value in points[0]]
    for _ in range(STEPS):
        numerators = [2 * numerator % denominator for numerator in numerators]
    assert np.array_equal(points[-1], np.array(numerators) / denominator)
    assert np.all(points[-1] > 0)
    _assert_uniform(points[-1])


def test_cubic_chaotic_map_walk_diffuses():
    """Test that the chaotic walk has no drift, a linear mean square displacement, and anticorrelated steps."""
    trajectory = (
        Simulation(
            built := cubic_chaotic_map(1000),
            dt=1,
            observables=describe('cubic_chaotic_map').observables_of(built),
        )
        .check()
        .prepare()
        .run(until=1000)
        .trajectory_
    )
    times = np.asarray(trajectory.times)
    square = np.asarray(trajectory.values('square_displacement'))
    slope, _ = np.polyfit(times[100:], square[100:], 1)
    assert slope > 0
    assert np.corrcoef(times[100:], square[100:])[0, 1] > 0.99
    assert np.all(np.abs(trajectory.values('displacement')[1:]) < STANDARD_DEVIATIONS * np.sqrt(square[1:] / 1000))
    steps = 2 * trajectory.values('walkers.x')[100:] - 1
    assert np.mean(steps[1:] * steps[:-1]) < -0.3 * np.mean(steps**2)


def test_cubic_chaotic_map_settles_below_chaos():
    """Test that the map settles on the fixed point 1/2 for λ = 5 and on a period-two cycle for λ = 10."""
    fixed = (
        Simulation(
            built := cubic_chaotic_map(3, lam=5.0),
            dt=1,
            observables=describe('cubic_chaotic_map').observables_of(built),
        )
        .check()
        .prepare()
        .run(until=400)
        .trajectory_.values('walkers.x')
    )
    cycle = (
        Simulation(
            built := cubic_chaotic_map(3, lam=10.0),
            dt=1,
            observables=describe('cubic_chaotic_map').observables_of(built),
        )
        .check()
        .prepare()
        .run(until=400)
        .trajectory_.values('walkers.x')
    )
    assert np.allclose(fixed[-1], 0.5)
    assert np.allclose(cycle[-1], cycle[-3])
    assert np.all(np.abs(cycle[-1] - cycle[-2]) > 0.1)
