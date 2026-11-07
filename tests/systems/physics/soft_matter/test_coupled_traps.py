"""Test the `_coupled_traps` module of `skcomplex.systems.physics.soft_matter`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system


def _free_steps(tensor, root='cholesky', separation=3.0):
    """Return the steps of both spheres while the traps are off, one row per step."""
    system = create_system('hydrodynamically_coupled_traps', separation=separation, tensor=tensor, root=root)
    trajectory = Simulation(system, dt=1, random_state=0).prepare().run(until=10000).trajectory_
    steps = trajectory.times[:-1] * 5e-3
    free = (steps % 0.4) >= 0.3 - 1e-9
    return np.diff(trajectory.values('spheres.x'), axis=0)[free]


@pytest.mark.parametrize('root', ['cholesky', 'lanczos'])
def test_free_spheres_move_together_as_the_mobility_predicts(root):
    """Test that released spheres correlate along and across their line of centres as `M_12 / M_11` of RPY."""
    steps = _free_steps('rotne-prager-yamakawa', root)
    along = 6 * math.pi * (1 / (4 * math.pi * 3.0) - 1 / (6 * math.pi * 27.0))
    across = 6 * math.pi * (1 + 2 / 27) / (8 * math.pi * 3.0)
    assert np.corrcoef(steps[:, 0, 0], steps[:, 1, 0])[0, 1] == pytest.approx(along, abs=0.06)
    assert np.corrcoef(steps[:, 0, 1], steps[:, 1, 1])[0, 1] == pytest.approx(across, abs=0.06)
    diffusion = 4.1e-3 / (6 * math.pi * 1e-3)
    assert steps[:, 0].var(axis=0) / (2 * diffusion * 5e-3) == pytest.approx([1.0, 1.0], rel=0.1)


def test_free_draining_spheres_move_independently():
    """Test that without hydrodynamic coupling the released spheres do not correlate."""
    steps = _free_steps('free-draining')
    assert abs(np.corrcoef(steps[:, 0, 0], steps[:, 1, 0])[0, 1]) < 0.06


def test_traps_hold_the_spheres_while_on():
    """Test that the spheres stay near their trap centres with the variance `kT / k` of the traps."""
    system = create_system('hydrodynamically_coupled_traps', off_time=0.0)
    positions = Simulation(system, dt=1, random_state=0).prepare().run(until=10000).trajectory_.values('spheres.x')
    assert positions[:, 0].mean(axis=0) == pytest.approx([-1.5, 0.0], abs=0.05)
    assert positions[:, 0].var(axis=0) == pytest.approx(np.full(2, 4.1e-3 / 0.2), rel=0.25)
