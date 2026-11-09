"""Test the `_cardiac` module of `skcomplex.systems.biology.physiology`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.biology.physiology import AlievPanfilovSystem, BarkleySystem


def test_aliev_panfilov_sustains_a_spiral():
    """Test that the broken wave in the Aliev-Panfilov tissue keeps the tissue active, as a re-entrant spiral."""
    trajectory = simulate(
        AlievPanfilovSystem((50, 50)).prepare().system_,
        until=400,
        dt=50.0,
        solver=RK4Solver(dt=0.05),
    )
    excited = (trajectory.get_values('tissue.u') > 0.5).mean(axis=1)
    assert np.all(excited[1:] > 0.05)


def test_barkley_sustains_its_spiral_and_breaks_up_in_the_chaotic_regime():
    """Test that the stripes of Barkley's medium keep it active, as a rotating spiral or as spatiotemporal chaos."""
    solver = RK4Solver(dt=0.2)
    spiral = simulate(BarkleySystem(), dt=20.0, solver=solver, until=500)
    chaos = simulate(BarkleySystem(a=0.7, epsilon=0.08, gamma=3.0).prepare().system_, until=500, dt=20.0, solver=solver)
    assert np.all((spiral.get_values('medium.u') > 0.5).mean(axis=1) > 0.01)
    assert np.all((chaos.get_values('medium.u')[10:] > 0.5).mean(axis=1) > 0.3)
