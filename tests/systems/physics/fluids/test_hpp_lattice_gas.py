"""Test the HPP gas of the `_lattice_gases` module of `skcomplex.systems.physics.fluids`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.physics.fluids import hpp_lattice_gas


def _channels(system, until):
    """Return the four channels of a run, as arrays of frames, rows, and columns."""
    trajectory = Simulation(system, dt=1).prepare().run(until=until).trajectory_
    shape = (-1, *system.elements[0].space.shape)
    return [np.asarray(trajectory.values(f'gas.n{direction}'), dtype=float).reshape(shape) for direction in range(4)]


def test_hpp_particle_moves_one_site_per_step_along_its_direction():
    """Test that a lone particle in each of two channels moves along its axis by one site per step."""
    system = hpp_lattice_gas((6, 6), density=0.0, bump=0.0)
    gas = system.elements[0]
    values = {f'n{direction}': np.zeros(36) for direction in range(4)}
    values['n0'][0], values['n1'][7] = 1.0, 1.0
    gas.set_params(values={**gas.values, **values, 'density': values['n0'] + values['n1']})
    east, north = _channels(system, 2)[:2]
    assert [np.flatnonzero(frame.ravel()).tolist() for frame in east] == [[0], [1], [2]]
    assert [np.flatnonzero(frame.ravel()).tolist() for frame in north] == [[7], [13], [19]]


@pytest.mark.parametrize('full', [False, True])
def test_hpp_conserves_particles_and_the_momentum_of_every_row_and_column(full):
    """Test that the particle number and the momentum along every row and every column stay fixed."""
    east, north, west, south = _channels(hpp_lattice_gas((32, 32), full=full), 40)
    totals = (east + north + west + south).sum(axis=(1, 2))
    rows, columns = (east - west).sum(axis=2), (north - south).sum(axis=1)
    assert np.all(totals == totals[0])
    assert np.all(rows == rows[0])
    assert np.all(columns == columns[0])


def test_hpp_full_cells_scatter_into_single_particles():
    """Test that full cells, which no collision changes, still break up as their particles move apart."""
    east, north, west, south = _channels(hpp_lattice_gas((32, 32), density=0.2, full=True), 20)
    full = east * north * west * south
    assert full[0].sum() > 0
    assert full[-1].sum() < full[0].sum()
