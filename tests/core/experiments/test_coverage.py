"""Test the `_coverage` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import CoverageResult, posterior_coverage
from skcomplex.core.experiments._coverage import _interval
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

TIMES = np.arange(0.0, 21.0, 2.0)


@pytest.mark.slow
def test_posterior_coverage_of_the_logistic_posterior_is_calibrated():
    """Test that MCMC on 40 prior draws covers the truths at levels 0.5 and 0.9 within 0.2."""
    result = posterior_coverage(
        LogisticGrowthSystem(n=10.0),
        {'r': (0.5, 1.5), 'K': (300.0, 800.0)},
        TIMES,
        n=40,
        levels=(0.5, 0.9),
        noise=10.0,
        n_samples=200,
        chains=2,
        random_state=0,
    )
    assert isinstance(result, CoverageResult)
    for level in (0.5, 0.9):
        for name in ('r', 'K'):
            assert abs(result.coverage[level][name] - level) <= 0.2
    assert result.ranks['r'].shape == (40,)
    assert np.all((result.ranks['K'] >= 0) & (result.ranks['K'] <= 1))
    assert len(result.results) == 40


@pytest.mark.slow
def test_posterior_coverage_runs_on_processes():
    """Test that variational coverage on processes gives the truths and ranks of a serial run."""
    settings = {
        'method': 'variational',
        'n': 2,
        'levels': (0.9,),
        'noise': 5.0,
        'n_samples': 50,
        'budget': 200,
        'random_state': 0,
    }
    bounds = {'r': (0.5, 1.0)}
    serial = posterior_coverage(LogisticGrowthSystem(n=10.0), bounds, TIMES, **settings)
    parallel = posterior_coverage(LogisticGrowthSystem(n=10.0), bounds, TIMES, executor='process', **settings)
    np.testing.assert_allclose(parallel.truths['r'], serial.truths['r'])
    np.testing.assert_allclose(parallel.ranks['r'], serial.ranks['r'])


def test_interval_of_weighted_samples():
    """Test the central interval of weighted samples."""
    values = np.arange(100.0)
    assert _interval(values, np.ones(100), 0.9) == (4.0, 94.0)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'method': 'local'}, 'posterior method'),
        ({'loss': 'events'}, 'takes the losses likelihood'),
        ({'levels': (1.5,)}, 'between 0 and 1'),
        ({'n': 0}, 'n = 0'),
        ({'executor': 'cluster'}, 'executor'),
        ({'executor': 'thread'}, 'cannot run on threads'),
    ],
    ids=['method', 'loss', 'level', 'n', 'executor', 'thread'],
)
def test_posterior_coverage_rejects_invalid_settings(arguments, message):
    """Test that bad methods, losses, levels, counts, and executors raise named errors."""
    with pytest.raises(ValidationError, match=message):
        posterior_coverage(LogisticGrowthSystem(), {'r': (0.5, 1.0)}, TIMES, **arguments)
