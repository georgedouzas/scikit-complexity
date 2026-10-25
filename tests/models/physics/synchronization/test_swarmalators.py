"""Test the `_swarmalators` module of `skcomplex.models.physics.synchronization`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.models.physics.synchronization import (
    swarmalators,
)


def _swarm_order(J, K):
    """Return the order parameters `S+`, `S-`, and the phase coherence of swarmalators at the end of a run."""
    trajectory = Simulation(swarmalators(60, J=J, K=K), dt=1.0, solver=RK4Solver(dt=0.1)).run(until=60).trajectory_
    positions = np.asarray(trajectory['units.x'])[-1]
    phases = np.asarray(trajectory['units.theta'])[-1]
    centred = positions - positions.mean(axis=0)
    angles = np.arctan2(centred[:, 1], centred[:, 0])
    return [abs(np.exp(1j * (angles + sign * phases)).mean()) for sign in (1, -1)] + [abs(np.exp(1j * phases).mean())]


def test_swarmalators_form_static_states():
    """Test the static phase wave at K = 0, the synchronized disk at K > 0, and the asynchronous disk at K < 0."""
    wave, sync, chaos = _swarm_order(1.0, 0.0), _swarm_order(1.0, 1.0), _swarm_order(0.1, -1.0)
    assert max(wave[:2]) > 0.9
    assert sync[2] > 0.99
    assert max(chaos) < 0.2
