"""Test the `_plastic_networks` module of `skcomplex.systems.biology.neuroscience`."""

import numpy as np

from skcomplex.systems.biology.neuroscience import AdaptiveDelayNetworkSystem, STDPNetworkSystem

from ..conftest import _run_until


def test_stdp_follows_the_integral_of_its_window():
    """Test that depression-dominated windows weaken the synapses and potentiation-dominated ones strengthen them."""
    weakened = _run_until(STDPNetworkSystem(A_minus=1.5), 150, dt=0.5).get_values('synapses.kappa')
    strengthened = _run_until(STDPNetworkSystem(A_minus=0.6), 150, dt=0.5).get_values('synapses.kappa')
    assert weakened[-1].mean() < 0.45
    assert strengthened[-1].mean() > 0.55
    assert np.all((weakened >= 0) & (strengthened <= 1))


def test_adaptive_delays_restore_synchrony():
    """Test that adaptive delays synchronize units that fixed delays leave incoherent."""

    def coherence(epsilon: float) -> float:
        trajectory = _run_until(AdaptiveDelayNetworkSystem(epsilon=epsilon), 300, dt=0.5)
        phases = np.array([trajectory.get_values(f'unit_{index}.theta') for index in range(5)]).T
        return float(np.abs(np.exp(1j * phases).mean(axis=1))[-200:].mean())

    assert coherence(0.1) > coherence(0.0) + 0.2
