"""Test the `_employment` module of `skcomplex.systems.social.organizations`."""

import numpy as np

from skcomplex.systems.social.organizations import (
    EmploymentTakeoverSystem,
)

from ..conftest import _run


def test_employment_takeover_is_deterministic_and_settles():
    """Test that the employment system gives the same run twice and ends in a fixed point or a short cycle."""
    first = _run(EmploymentTakeoverSystem(), 60, dt=1).get_values('cells.colour')
    second = _run(EmploymentTakeoverSystem(), 60, dt=1).get_values('cells.colour')
    assert np.array_equal(first, second)
    assert any(np.array_equal(first[-1], first[-1 - period]) for period in (1, 2))
