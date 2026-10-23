"""Test the `_lattice_gases` module of `skcomplex.models.physics.fluids`."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.models.physics.fluids import fhp_lattice_gas


def test_fhp_lattice_gas_conserves_mass_and_momentum():
    """Test that the FHP gas on a periodic 64 x 64 lattice keeps its particle number and momentum for 500 steps."""
    trajectory = Simulation(fhp_lattice_gas((64, 64)), dt=1, random_state=0).run(until=500).trajectory_
    occupations = np.stack([trajectory[f'gas.n{direction}'].astype(float).sum(axis=1) for direction in range(6)])
    angles = np.arange(6) * np.pi / 3
    momentum = np.stack([np.cos(angles) @ occupations, np.sin(angles) @ occupations])
    assert len(trajectory) == 501
    assert np.all(occupations.sum(axis=0) == occupations.sum(axis=0)[0])
    assert np.allclose(momentum, momentum[:, :1], rtol=0, atol=1e-9)
    assert np.array_equal(trajectory['gas.density'].sum(axis=1), occupations.sum(axis=0))
