"""Test the `_lattice_gases` module of `skcomplex.systems.physics.fluids`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.fluids import FHPLatticeGasSystem


def test_fhp_lattice_gas_conserves_mass_and_momentum():
    """Test that the FHP gas on a periodic 64 x 64 lattice keeps its particle number and momentum for 500 steps."""
    trajectory = simulate(FHPLatticeGasSystem((64, 64)), until=500, random_state=0)
    occupations = np.stack([trajectory.values(f'gas.n{direction}').astype(float).sum(axis=1) for direction in range(6)])
    angles = np.arange(6) * np.pi / 3
    momentum = np.stack([np.cos(angles) @ occupations, np.sin(angles) @ occupations])
    assert len(trajectory.times) == 501
    assert np.all(occupations.sum(axis=0) == occupations.sum(axis=0)[0])
    assert np.allclose(momentum, momentum[:, :1], rtol=0, atol=1e-9)
    assert np.array_equal(trajectory.values('gas.density').sum(axis=1), occupations.sum(axis=0))
