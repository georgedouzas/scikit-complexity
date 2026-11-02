"""Test the `_low_order` module of `skcomplex.models.physics.climate`."""

import numpy as np

from skcomplex.models.physics.climate import (
    geodynamo,
    lorenz_84,
    lorenz_96,
)

from ..conftest import _run_default
from .conftest import _sign_changes


def test_lorenz_84_relaxes_to_the_hadley_circulation():
    """Test that without eddies or land-sea forcing the jet relaxes to the forcing `F`."""
    jet = _run_default(lorenz_84(y=0.0, z=0.0, G=0.0), until=60.0, dt=1.0).values('atmosphere.x')[-1]
    assert abs(jet - 8.0) < 1e-3


def test_lorenz_96_steady_state_is_unstable():
    """Test that the uniform state `F` of Lorenz-96 is kept exactly and that a small perturbation of it grows."""
    steady = _run_default(lorenz_96(8, perturbation=0.0), until=2.0, dt=0.5).values('ring.x')
    assert np.allclose(steady, 8.0)
    perturbed = _run_default(lorenz_96(8), until=5.0, dt=0.5).values('ring.x')
    assert np.abs(perturbed[-1] - 8.0).max() > 1.0


def test_geodynamo_is_symmetric_under_polarity_reversal():
    """Test that reversing both fields reverses the trajectory, and that the dipole reverses on its own."""
    north = _run_default(geodynamo(Q=0.1, D=0.2), until=50.0, dt=1.0)
    south = _run_default(geodynamo(Q=-0.1, D=-0.2), until=50.0, dt=1.0)
    assert np.allclose(north.values('dynamo.D'), -south.values('dynamo.D'), atol=1e-4)
    assert np.allclose(north.values('dynamo.V'), south.values('dynamo.V'), atol=1e-4)
    assert _sign_changes(_run_default(geodynamo(), until=300.0, dt=1.0).values('dynamo.D')) > 2
