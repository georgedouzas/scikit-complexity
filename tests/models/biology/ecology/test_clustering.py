"""Test the `_clustering` module of `skcomplex.models.biology.ecology`."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.engine.backends import NumbaBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.biology.ecology import neutral_clustering


def _expected_sample_variance(sites, alpha, beta, n, times):
    """Return the expected variance over the sites of a ring, from the exact moment equations of neutral clustering."""

    def derivative(t, covariance):
        rates = 2 * beta * (np.roll(covariance, 1) + np.roll(covariance, -1) - 2 * covariance)
        rates[0] += 2 * alpha * n
        return rates

    initial = np.zeros(sites)
    initial[0] = -n
    solution = solve_ivp(derivative, (0, times[-1]), initial, t_eval=times, rtol=1e-8).y
    covariance = solution + n * (np.arange(sites) == 0)[:, None]
    return covariance[0] - covariance.mean(axis=0)


def test_neutral_clustering_variance_grows_linearly_without_hops():
    """Test that without hops the variance over 400 patches, averaged over six runs, grows as 2 alpha n t."""
    system = neutral_clustering(400, n=10.0, alpha=1.0, beta=0.0)
    runs = [
        Simulation(
            system,
            dt=2.0,
            random_state=seed,
            backend=NumbaBackend(),
            observables=describe('neutral_clustering').observables_of(system),
        ).check()
        .run(until=10)
        .trajectory_
        for seed in range(6)
    ]
    times = runs[0].times
    assert np.mean([run.values('mean')[-1] for run in runs]) == pytest.approx(10.0, rel=0.1)
    assert np.mean([run.values('variance') for run in runs], axis=0)[1:] == pytest.approx(2 * 10.0 * times[1:], rel=0.2)


def test_neutral_clustering_hops_slow_but_keep_the_clustering():
    """Test that hops on a ring give the variance of the exact moment equations, which grows below 2 alpha n t."""
    sites, until = 64, 20.0
    system = neutral_clustering(sites, n=10.0, alpha=1.0, beta=1.0)
    variances = [
        Simulation(
            system,
            dt=5.0,
            random_state=seed,
            backend=NumbaBackend(),
            observables=describe('neutral_clustering').observables_of(system),
        ).check()
        .run(until=until)
        .trajectory_.values('variance')
        for seed in range(32)
    ]
    times = np.arange(0.0, until + 1, 5.0)
    expected = _expected_sample_variance(sites, 1.0, 1.0, 10.0, times)
    assert np.mean(variances, axis=0)[1:] == pytest.approx(expected[1:], rel=0.25)
    assert expected[-1] < 0.5 * 2 * 10.0 * until
    assert expected[-1] > 1.5 * expected[1]
