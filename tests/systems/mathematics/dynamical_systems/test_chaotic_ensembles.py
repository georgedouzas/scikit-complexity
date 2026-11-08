"""Test the `_chaotic_ensembles` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import math

import numpy as np
import pytest

from skcomplex.core.analysis import displacement
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.mathematics.dynamical_systems import (
    CubicChaoticMapSystem,
    DoublingMapSystem,
    SkewTentMapSystem,
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
    system = SkewTentMapSystem(ENSEMBLE_SIZE, a=a, random_state=1)
    points = simulate(system, until=STEPS, dt=1).values('points.x')
    assert np.ptp(points[0]) <= 0.01
    _assert_uniform(points[-1])


def test_doubling_map_iterates_fractions_exactly():
    """Test that the doubling map follows `k' = 2k mod q` exactly well beyond the floating-point collapse."""
    system = DoublingMapSystem(ENSEMBLE_SIZE, random_state=2).prepare().system_
    denominator = int(system.elements[0].values['q'])
    points = simulate(system, until=STEPS, dt=1).values('points.x')
    numerators = [round(value * denominator) for value in points[0]]
    for _ in range(STEPS):
        numerators = [2 * numerator % denominator for numerator in numerators]
    assert np.array_equal(points[-1], np.array(numerators) / denominator)
    assert np.all(points[-1] > 0)
    _assert_uniform(points[-1])


def test_cubic_chaotic_map_walk_diffuses():
    """Test that the chaotic walk has no drift, a linear mean square displacement, and anticorrelated steps."""
    trajectory = simulate(
        CubicChaoticMapSystem(1000),
        until=1000,
        dt=1,
    )
    times = np.asarray(trajectory.times)
    square = np.asarray(trajectory.values('square_displacement'))
    slope, _ = np.polyfit(times[100:], square[100:], 1)
    assert slope > 0
    assert np.corrcoef(times[100:], square[100:])[0, 1] > 0.99
    assert np.all(np.abs(trajectory.values('displacement')[1:]) < STANDARD_DEVIATIONS * np.sqrt(square[1:] / 1000))
    steps = 2 * trajectory.values('walkers.x')[100:] - 1
    assert np.mean(steps[1:] * steps[:-1]) < -0.3 * np.mean(steps**2)


def test_cubic_chaotic_map_displacement_reads_the_walker():
    """Test that the displacement of the system reads the walker `y`, not the map value `x`, and spreads."""
    trajectory = simulate(CubicChaoticMapSystem(), until=500, random_state=0)
    analysis = displacement(trajectory)
    np.testing.assert_array_equal(analysis.positions[..., 0], trajectory.values('walkers.y'))
    assert analysis.exponent > 0.8


def test_cubic_chaotic_map_settles_below_chaos():
    """Test that the map settles on the fixed point 1/2 for λ = 5 and on a period-two cycle for λ = 10."""
    fixed = simulate(
        CubicChaoticMapSystem(3, lam=5.0),
        until=400,
        dt=1,
    ).values('walkers.x')
    cycle = simulate(
        CubicChaoticMapSystem(3, lam=10.0),
        until=400,
        dt=1,
    ).values('walkers.x')
    assert np.allclose(fixed[-1], 0.5)
    assert np.allclose(cycle[-1], cycle[-3])
    assert np.all(np.abs(cycle[-1] - cycle[-2]) > 0.1)
