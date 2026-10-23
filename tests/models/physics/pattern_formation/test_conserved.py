"""Test the `_conserved` module of `skcomplex.models.physics.pattern_formation`."""

import numpy as np

from skcomplex.core.analysis import dispersion_relation, fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.models.physics.pattern_formation import (
    nonreciprocal_cahn_hilliard,
)


def _domain_speed(**coupling):
    """Return the speed of the domains of the nonreciprocal Cahn-Hilliard model over the second half of a run."""
    length = 40.0
    system = nonreciprocal_cahn_hilliard((64,), length=length, **coupling)
    phi = fields_of(Simulation(system, dt=2.0).run(until=300).trajectory_, 'mixture.phi')
    modes = np.fft.rfft(phi, axis=1)
    mode = 1 + np.argmax(np.abs(modes[-1, 1:]))
    phases = np.unwrap(np.angle(modes[75:, mode]))
    drift = np.polyfit(2.0 * np.arange(75, len(phi)), phases, 1)[0]
    return abs(drift) / (2 * np.pi * mode / length)


def test_nonreciprocal_cahn_hilliard_domains_travel_above_threshold():
    """Test that the demixed domains travel when alpha^2 - chi^2 exceeds D^2, and stay put otherwise."""
    assert _domain_speed(alpha=0.3) > 0.05
    assert _domain_speed(alpha=0.05) < 0.005
    assert _domain_speed(chi=0.3, alpha=0.0) < 0.005


def test_nonreciprocal_cahn_hilliard_dispersion_relation():
    """Test that the unstable band propagates above threshold, on the band edge of Brauns and Marchetti."""
    for alpha, travels in [(0.3, True), (0.05, False)]:
        relation = dispersion_relation(nonreciprocal_cahn_hilliard((256,), length=200.0, alpha=alpha))
        growing = relation.growth_rates > 1e-9
        assert (np.abs(relation.frequencies[growing]).max() > 1e-6) == travels
        edge = np.sqrt(1 - 0.1) if travels else np.sqrt(1 - alpha**2 / 0.1)
        np.testing.assert_allclose(relation.magnitudes[growing].max(), edge, atol=2 * np.pi / 200.0)
