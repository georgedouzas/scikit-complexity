"""Test the `_spring_blocks` module of `skcomplex.models.earth.seismology`."""

import warnings

import numpy as np

from skcomplex.models import describe
from skcomplex.models.earth.seismology import burridge_knopoff

from ..conftest import _run


def test_burridge_knopoff_sticks_and_slips():
    """Test that the blocks never slide backwards, slip in bursts, and follow the plate."""
    trajectory = _run(burridge_knopoff(velocity=0.05), until=40.0, dt=0.5)
    x, v = trajectory['fault.x'], trajectory['fault.v']
    assert np.all(np.diff(x, axis=0) > -1e-9)
    assert (v > 0.01).any()
    assert (np.abs(v) < 1e-6).mean() > 0.5
    assert x[-1].mean() > 0.05 * 40.0 - 2.0


def test_burridge_knopoff_default_solver_does_not_stall():
    """Test that the registered solver runs the fault through its slips without the stall of LSODA."""
    solver = describe('burridge_knopoff').defaults['solver']
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        trajectory = _run(burridge_knopoff(), until=400.0, dt=10.0, solver=solver)
    steps = np.diff(trajectory['fault.x'], axis=0)
    assert steps.min() > -1e-9
    assert (steps > 0.3).any()
