"""Test the `_employment` module of `skcomplex.systems.social.organizations`."""

import numpy as np

from skcomplex.systems.social.organizations import (
    employment_takeover,
)

from ..conftest import _run


def test_employment_takeover_is_deterministic_and_settles():
    """Test that the employment system gives the same run twice and ends in a fixed point or a short cycle."""
    first = _run(employment_takeover(), 60, dt=1).values('cells.colour')
    second = _run(employment_takeover(), 60, dt=1).values('cells.colour')
    assert np.array_equal(first, second)
    assert any(np.array_equal(first[-1], first[-1 - period]) for period in (1, 2))
