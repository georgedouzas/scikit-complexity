"""Test the `_disease` module of `skcomplex.systems.biology.physiology`."""

import math

import numpy as np

from skcomplex.systems.biology.physiology import two_layer_sepsis

from ..conftest import _run_until


def _frequency_spread(beta: float) -> float:
    """Return the spread of the mean frequencies of the parenchymal cells over the last 100 time units."""
    trajectory = _run_until(two_layer_sepsis(beta=beta), 300, dt=1.0)
    phases = np.unwrap(trajectory.values('cells.phi'), axis=0)
    frequencies = (phases[-1] - phases[-101]) / 100
    return float(frequencies[:20].std())


def test_the_age_parameter_turns_the_healthy_state_pathological():
    """Test that the layers lock their frequencies at a low age parameter and lose synchrony at `beta = pi / 2`."""
    assert _frequency_spread(0.25 * math.pi) < 0.02
    assert _frequency_spread(0.5 * math.pi) > 0.1


def test_adaptive_couplings_stay_bounded():
    """Test that the cytokine couplings stay in `[-1, 1]`, where the adaptation keeps them."""
    trajectory = _run_until(two_layer_sepsis(), 100, dt=1.0)
    for links in ('parenchymal_links', 'immune_links'):
        assert np.abs(trajectory.values(f'{links}.kappa')).max() <= 1.0 + 1e-6
