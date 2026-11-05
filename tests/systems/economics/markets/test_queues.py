"""Test the `_queues` module of `skcomplex.systems.economics.markets`."""

import numpy as np

from skcomplex.systems.economics.markets import (
    car_wash,
)

from ..conftest import _agents, _run


def test_car_wash_books_every_payment():
    """Test that the revenue prices every wash and valet, and the wash queue never exceeds the balking length."""
    trajectory = _run(car_wash(p_enter=0.05), 3600)
    till = _agents(trajectory, 'till', 3600)
    assert till['revenue'][0] == 4.0 * till['washes'][0] + 10.0 * till['valets'][0]
    assert till['washes'][0] + till['valets'][0] > 0
    assert till['turned_away'][0] > 0
    for frame in range(0, 3601, 100):
        assert np.sum(_agents(trajectory, 'cars', frame)['stage'] == 0) <= 3


def test_car_wash_runs_reproducibly():
    """Test that two runs of the car wash with the same random state take the same revenue."""
    first = _agents(_run(car_wash(p_enter=0.05), 600), 'till', 600)['revenue']
    second = _agents(_run(car_wash(p_enter=0.05), 600), 'till', 600)['revenue']
    assert np.array_equal(first, second)
