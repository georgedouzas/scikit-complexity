"""Test the `_diffusion` module of `skcomplex.models.social.opinion_dynamics`."""

import numpy as np
import pytest

from skcomplex.core.base import ValidationError
from skcomplex.models.social.opinion_dynamics import (
    influentials,
    threshold_adoption,
)

from ..conftest import _run, _values


def test_influentials_add_their_weight():
    """Test that beliefs grow by eight neighbours per agent and influentials pull agents to caring passionately."""
    plain = _run(influentials(n_influentials=0, random_state=0), 20)
    total = [_values(plain, 'people', 'belief', frame).sum() for frame in (0, 1)]
    assert total[1] - total[0] == 8 * 400
    swayed = _run(influentials(n_influentials=40, random_state=0), 20)
    passionate = [(_values(trajectory, 'people', 'caring') == 3).mean() for trajectory in (plain, swayed)]
    assert passionate[1] > passionate[0]


def test_threshold_adoption_by_infection_spreads_to_everyone():
    """Test that adoption by infection with a reach of 15 reaches everyone and never decreases."""
    trajectory = _run(threshold_adoption(random_state=0), 25)
    shares = [_values(trajectory, 'people', 'adopted', frame).mean() for frame in range(26)]
    assert shares[-1] == 1.0
    assert np.all(np.diff(shares) >= 0)


def test_threshold_adoption_rejects_unknown_mode():
    """Test that an unknown adoption mode raises a validation error when the model compiles."""
    with pytest.raises(ValidationError, match='Unknown adoption mode'):
        _run(threshold_adoption(40, size=60.0, mode='contagion'), 1)
