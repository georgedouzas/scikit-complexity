"""Test the `_commons` module of `skcomplex.models.economics.resources`."""

import numpy as np

from skcomplex.models import run


def test_commons_collapse():
    """Test that an unregulated meadow with too many herders falls below a tenth of its grass and the herd shrinks."""
    trajectory = run('tragedy_of_the_commons', settings={'until': 60}).trajectory_
    capacity = trajectory['grass'][0]
    collapse = int(np.argmax(trajectory['grass'] < 0.1 * capacity))
    assert trajectory['grass'][collapse] < 0.1 * capacity
    herd = trajectory['herd']
    assert herd[collapse:].min() < 0.5 * herd[: collapse + 1].max()


def test_regulated_commons_keeps_its_grass():
    """Test that a regulator with an adaptive quota keeps the grass above a tenth of its capacity."""
    trajectory = run('tragedy_of_the_commons', quota=2.0, settings={'until': 60}).trajectory_
    assert trajectory['grass'].min() > 0.1 * trajectory['grass'][0]
    assert np.diff(trajectory['quota']).any()
