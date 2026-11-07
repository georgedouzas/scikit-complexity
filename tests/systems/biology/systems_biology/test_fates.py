"""Test the `_fates` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np

from skcomplex.systems.biology.systems_biology import BristleLateralInhibitionSystem, ThreeWayFateDecisionSystem

from ..conftest import _run_until


def _fates(trajectory) -> np.ndarray:
    """Return the distinct final states of the cells, rounded."""
    final = np.c_[trajectory.values('cells.x')[-1], trajectory.values('cells.y')[-1]]
    return np.unique(np.round(final, 2), axis=0)


def test_cells_split_among_three_fates_until_a_signal_removes_two():
    """Test the elliptic umbilic: three attractors without signal, and one after a signal crosses the fold curves."""
    assert len(_fates(_run_until(ThreeWayFateDecisionSystem(), 40, dt=0.5))) == 3
    assert len(_fates(_run_until(ThreeWayFateDecisionSystem(v1=0.05), 40, dt=0.5))) == 1


def test_lateral_inhibition_leaves_spaced_neural_cells():
    """Test that the ring ends with binary fates and neural cells at least four cell diameters apart."""
    for seed in range(3):
        states = _run_until(BristleLateralInhibitionSystem(seed=seed), 60, dt=1.0).values('cells.a')[-1]
        assert np.all((states < 0.05) | (states > 0.95))
        neural = np.flatnonzero(states > 0.5)
        assert np.diff(np.r_[neural, neural[0] + states.size]).min() >= 4
