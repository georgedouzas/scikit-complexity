"""Test the `_commons` module of `skcomplex.systems.economics.resources`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.resources import TragedyOfTheCommonsSystem


def test_commons_collapse():
    """Test that an unregulated meadow with too many herders falls below a tenth of its grass and the herd shrinks."""
    trajectory = simulate(TragedyOfTheCommonsSystem(), until=60)
    capacity = trajectory.values('grass')[0]
    collapse = int(np.argmax(trajectory.values('grass') < 0.1 * capacity))
    assert trajectory.values('grass')[collapse] < 0.1 * capacity
    herd = trajectory.values('herd')
    assert herd[collapse:].min() < 0.5 * herd[: collapse + 1].max()


def test_regulated_commons_keeps_its_grass():
    """Test that a regulator with an adaptive quota keeps the grass above a tenth of its capacity."""
    trajectory = simulate(TragedyOfTheCommonsSystem(quota=2.0), until=60)
    assert trajectory.values('grass').min() > 0.1 * trajectory.values('grass')[0]
    assert np.diff(trajectory.values('quota')).any()
