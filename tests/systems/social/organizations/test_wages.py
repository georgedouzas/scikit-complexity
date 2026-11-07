"""Test the `_wages` module of `skcomplex.systems.social.organizations`."""

import numpy as np

from skcomplex.systems.social.organizations import AjanisChoiceSystem

from ..conftest import _run, _values


def test_ajanis_choice_pays_below_the_minimum_only_when_workers_are_plentiful():
    """Test that wages fall below the minimum and slaves appear when workers outnumber jobs, and not otherwise."""
    plentiful = _run(AjanisChoiceSystem(), 60)
    assert plentiful.values('day.slaves')[1:].sum() > 0
    assert np.min([np.mean(_values(plentiful, 'employers', 'wage', frame)) for frame in range(61)]) < 1.0
    scarce = _run(AjanisChoiceSystem(n_workers=300), 60)
    assert scarce.values('day.slaves')[1:].sum() == 0
    assert np.mean(_values(scarce, 'employers', 'wage')) > 1.5


def test_ajanis_choice_keeps_hires_within_the_workforce():
    """Test that no employer hires more workers than it needs."""
    trajectory = _run(AjanisChoiceSystem(), 10)
    for frame in range(1, 11):
        employed = _values(trajectory, 'employers', 'employed', frame)
        assert np.all(employed <= _values(trajectory, 'employers', 'workforce', frame))
        assert employed.sum() == _values(trajectory, 'workers', 'working', frame).sum()
