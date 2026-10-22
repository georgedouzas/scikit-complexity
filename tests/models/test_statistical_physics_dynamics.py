"""Test the Bertin maps and coupled oscillators: chaotic ensembles, the subcritical Hopf form, and amplitude death."""

import math

import numpy as np
import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.mathematics.dynamical_systems import (
    cubic_chaotic_map,
    doubling_map,
    skew_tent_map,
    subcritical_hopf_form,
)
from skcomplex.models.physics.synchronization import heterogeneous_global_coupling, oscillator_death

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


def _decay_rate(times, values, start):
    """Return the slope of the logarithm of positive values after a start time."""
    selected = times >= start
    return np.polyfit(times[selected], np.log(values[selected]), 1)[0]


@pytest.mark.parametrize('a', [0.3, 0.8])
def test_skew_tent_map_invariant_density_is_uniform(a):
    """Test that 10000 points iterated 200 steps fill every one of 20 bins within four binomial deviations."""
    system = skew_tent_map(ENSEMBLE_SIZE, a=a, random_state=1)
    points = Simulation(system, dt=1).run(until=STEPS).trajectory_['points.x']
    assert np.ptp(points[0]) <= 0.01
    _assert_uniform(points[-1])


def test_doubling_map_iterates_fractions_exactly():
    """Test that the doubling map follows `k' = 2k mod q` exactly well beyond the floating-point collapse."""
    system = doubling_map(ENSEMBLE_SIZE, random_state=2)
    denominator = int(system.elements[0].values['q'])
    points = Simulation(system, dt=1).run(until=STEPS).trajectory_['points.x']
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
        .run(until=1000)
        .trajectory_
    )
    times = np.asarray(trajectory.times)
    square = np.asarray(trajectory['square_displacement'])
    slope, _ = np.polyfit(times[100:], square[100:], 1)
    assert slope > 0
    assert np.corrcoef(times[100:], square[100:])[0, 1] > 0.99
    assert np.all(np.abs(trajectory['displacement'][1:]) < STANDARD_DEVIATIONS * np.sqrt(square[1:] / 1000))
    steps = 2 * trajectory['walkers.x'][100:] - 1
    assert np.mean(steps[1:] * steps[:-1]) < -0.3 * np.mean(steps**2)


def test_cubic_chaotic_map_settles_below_chaos():
    """Test that the map settles on the fixed point 1/2 for λ = 5 and on a period-two cycle for λ = 10."""
    fixed = (
        Simulation(
            built := cubic_chaotic_map(3, lam=5.0),
            dt=1,
            observables=describe('cubic_chaotic_map').observables_of(built),
        )
        .run(until=400)
        .trajectory_['walkers.x']
    )
    cycle = (
        Simulation(
            built := cubic_chaotic_map(3, lam=10.0),
            dt=1,
            observables=describe('cubic_chaotic_map').observables_of(built),
        )
        .run(until=400)
        .trajectory_['walkers.x']
    )
    assert np.allclose(fixed[-1], 0.5)
    assert np.allclose(cycle[-1], cycle[-3])
    assert np.all(np.abs(cycle[-1] - cycle[-2]) > 0.1)


def test_subcritical_hopf_form_is_bistable():
    """Test that the focus and the cycle of radius `r² = (1 + √(1 + 4 alpha)) / 2` attract small and large starts."""
    alpha = -0.1
    radius = math.sqrt((1 + math.sqrt(1 + 4 * alpha)) / 2)
    small = Simulation(subcritical_hopf_form(0.1, alpha=alpha), dt=0.1).run(until=100).state_
    large = Simulation(subcritical_hopf_form(1.0, alpha=alpha), dt=0.1).run(until=100).state_
    assert np.hypot(*small) < 1e-3
    assert np.hypot(*large) == pytest.approx(radius, abs=1e-4)
    gone = Simulation(subcritical_hopf_form(1.0, alpha=-0.3), dt=0.1).run(until=100).state_
    assert np.hypot(*gone) < 1e-3


def test_subcritical_hopf_form_shows_hysteresis():
    """Test that the amplitude at alpha = -0.1 depends on whether alpha came from below the fold or from above 0."""
    simulation = Simulation(subcritical_hopf_form(0.01, alpha=-0.1), dt=0.1)
    before = np.hypot(*simulation.run(until=50).state_)
    simulation.update_parameters({'form.alpha': 0.2})
    simulation.extend(until=150)
    simulation.update_parameters({'form.alpha': -0.1})
    after = np.hypot(*simulation.extend(until=250).state_)
    simulation.update_parameters({'form.alpha': -0.3})
    fallen = np.hypot(*simulation.extend(until=350).state_)
    assert before < 1e-3
    assert after > 0.9
    assert fallen < 1e-3


def test_oscillator_death_follows_lorentzian_threshold():
    """Test that oscillations die at the rate `ω0 - 1` above the critical width 1 and persist below it."""
    dying = (
        Simulation(
            built := oscillator_death(100, K=3.0, width=1.5),
            dt=0.5,
            observables=describe('oscillator_death').observables_of(built),
        )
        .run(until=20)
        .trajectory_
    )
    amplitude = np.asarray(dying['amplitude'])
    rate = _decay_rate(np.asarray(dying.times), amplitude, start=8.0)
    assert amplitude[-1] < 1e-3
    assert rate == pytest.approx(1 - 1.5, rel=0.1)
    living = (
        Simulation(
            built := oscillator_death(100, K=3.0, width=0.5),
            dt=0.5,
            observables=describe('oscillator_death').observables_of(built),
        )
        .run(until=20)
        .trajectory_
    )
    assert living['amplitude'][-1] > 0.3


def test_oscillator_death_follows_quartic_threshold():
    """Test that the quartic density kills the oscillations above `(1 + √(2K - 1)) / √2` and not below it."""
    critical = (1 + math.sqrt(2 * 2.0 - 1)) / math.sqrt(2)
    above = oscillator_death(100, K=2.0, width=1.2 * critical, distribution='quartic')
    below = oscillator_death(100, K=2.0, width=0.8 * critical, distribution='quartic')
    assert (
        Simulation(above, dt=0.5, observables=describe('oscillator_death').observables_of(above))
        .run(until=40)
        .trajectory_['amplitude'][-1]
        < 1e-3
    )
    assert (
        Simulation(below, dt=0.5, observables=describe('oscillator_death').observables_of(below))
        .run(until=40)
        .trajectory_['amplitude'][-1]
        > 0.3
    )


def test_oscillator_death_rejects_unknown_distribution():
    """Test that an unknown density of frequencies raises a validation error that names the choices."""
    with pytest.raises(ValidationError, match='lorentzian'):
        oscillator_death(10, distribution='gaussian')


def test_heterogeneous_global_coupling_restabilizes_fixed_point():
    """Test that intermediate coupling kills the oscillations, while weak and very strong coupling do not."""
    amplitudes = {
        k: Simulation(
            built := heterogeneous_global_coupling(100, k=k),
            dt=0.5,
            observables=describe('heterogeneous_global_coupling').observables_of(built),
        )
        .run(until=60)
        .trajectory_['amplitude'][-1]
        for k in (0.0, 2.0, 15.0)
    }
    assert amplitudes[0.0] > 0.1
    assert amplitudes[2.0] < 1e-3
    assert amplitudes[15.0] > 0.01


def test_heterogeneous_global_coupling_decays_as_order_parameters_predict():
    """Test that the decay rate at k = 4 matches the leading eigenvalue of the closed order-parameter equations."""
    system = heterogeneous_global_coupling(100, k=4.0)
    taus = system.elements[0].values['tau']
    nu, variance = complex(0.1, 2.0), float(np.var(taus))
    roots = [nu - 2.0 + sign * np.sqrt(4.0 + variance * nu**2) for sign in (1, -1)]
    predicted = max(root.real for root in roots)
    trajectory = (
        Simulation(system, dt=0.5, observables=describe('heterogeneous_global_coupling').observables_of(system))
        .run(until=60)
        .trajectory_
    )
    rate = _decay_rate(np.asarray(trajectory.times), np.asarray(trajectory['amplitude']), start=20.0)
    assert predicted < 0
    assert rate == pytest.approx(predicted, rel=0.15)
