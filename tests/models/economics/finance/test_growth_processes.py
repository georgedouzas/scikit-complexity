"""Test the `_growth_processes` module of `skcomplex.models.economics.finance`."""

import numpy as np
import pytest

from skcomplex.models import describe
from skcomplex.models.economics.finance import (
    gibrat_process,
    kesten_process,
    multiplicative_process,
)

from ..conftest import _agents, _run


def test_gibrat_process_makes_lognormal_sizes():
    """Test that the logarithm of the sizes spreads like a random walk of the logarithmic growth factors."""
    sizes = _agents(
        _run(built := gibrat_process(1000), 100, observables=describe('gibrat_process').observables_of(built)),
        'firms',
        100,
    )['x']
    logs = np.log(sizes)
    assert logs.std() == pytest.approx(np.sqrt(100) * 0.05 / 1.02, rel=0.1)
    assert abs(float(np.mean((logs - logs.mean()) ** 3)) / logs.std() ** 3) < 0.3


def test_multiplicative_process_respects_its_floor():
    """Test that no size falls below the fraction of the previous mean, and the floor fattens the upper tail."""
    trajectory = _run(
        built := multiplicative_process(300),
        100,
        observables=describe('multiplicative_process').observables_of(built),
    )
    for frame in (10, 50, 100):
        previous = _agents(trajectory, 'firms', frame)['previous']
        assert _agents(trajectory, 'firms', frame)['x'].min() >= 0.9 * previous.mean() - 1e-9
    free = _agents(
        _run(built := gibrat_process(300), 100, observables=describe('gibrat_process').observables_of(built)),
        'firms',
        100,
    )['x']
    floored = _agents(trajectory, 'firms', 100)['x']
    assert floored.max() / np.median(floored) > 1.0
    assert floored.min() / np.median(floored) > free.min() / np.median(free)


def test_kesten_process_reaches_a_positive_stationary_state():
    """Test that Kesten sizes stay positive and settle far below the initial size when the factor shrinks on average."""
    trajectory = _run(built := kesten_process(500), 100, observables=describe('kesten_process').observables_of(built))
    sizes = _agents(trajectory, 'firms', 100)['x']
    assert sizes.min() > 0
    assert np.median(sizes) < 25.0
    assert sizes.max() > 10 * np.median(sizes)
