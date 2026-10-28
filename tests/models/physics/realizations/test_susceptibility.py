"""Test the `_susceptibility` module of `skcomplex.models.physics.realizations`."""

import math

import numpy as np
import pytest

from skcomplex.analysis import DistributionAnalysis
from skcomplex.models.physics.realizations import SusceptibilityAnalysis

from ....analysis.conftest import ISING_TEMPERATURES, _magnetizations

CRITICAL = 2 / math.log(1 + math.sqrt(2))


def _kuramoto_order(rng: np.random.Generator, coupling: float, oscillators: int) -> np.ndarray:
    """Return the order parameter of Kuramoto oscillators with Lorentzian frequencies of width 1 after relaxation."""
    frequencies = np.tan(np.pi * (rng.random(oscillators) - 0.5))
    phases = rng.uniform(-np.pi, np.pi, oscillators)
    orders = []
    for step in range(6000):
        order = np.mean(np.exp(1j * phases))
        phases = phases + 0.05 * (frequencies + coupling * np.abs(order) * np.sin(np.angle(order) - phases))
        if step >= 2000:
            orders.append(np.abs(order))
    return np.asarray(orders)


def test_susceptibility_scales_the_variance_of_absolute_samples():
    """Test that the susceptibility is the variance of the absolute samples times the size over the temperature."""
    samples = np.random.default_rng(0).standard_normal((500, 5)) * 0.3 + 0.2
    temperatures = np.linspace(1.0, 2.0, 5)
    susceptibility = SusceptibilityAnalysis(size=64, temperatures=temperatures).check().prepare(samples).run()
    general = DistributionAnalysis('moments', absolute=True).check().prepare(samples).run()
    for name in ('mean_', 'variance_', 'skewness_', 'kurtosis_'):
        np.testing.assert_array_equal(getattr(susceptibility, name), getattr(general, name))
    np.testing.assert_allclose(susceptibility.susceptibility_, 64 * general.variance_ / temperatures, rtol=1e-15)
    assert susceptibility.peak_ is None
    with pytest.raises(Exception, match='positive size'):
        SusceptibilityAnalysis(size=0).check()


@pytest.mark.slow
def test_susceptibility_peaks_at_two_transitions():
    """Test that the susceptibility peaks of a 64 by 64 Ising lattice and of Kuramoto oscillators locate T_c and K_c."""
    rng = np.random.default_rng(0)
    samples = np.column_stack([_magnetizations(64, temperature, rng) for temperature in ISING_TEMPERATURES])
    ising = SusceptibilityAnalysis(size=64**2, temperatures=ISING_TEMPERATURES, parameters=ISING_TEMPERATURES)
    assert ising.check().prepare(samples).run().peak_ == pytest.approx(CRITICAL, rel=0.03)
    couplings = np.linspace(1.0, 3.0, 11)
    rng = np.random.default_rng(4)
    orders = np.column_stack([_kuramoto_order(rng, coupling, 2000) for coupling in couplings])
    kuramoto = SusceptibilityAnalysis(size=2000, parameters=couplings).check().prepare(orders).run()
    assert kuramoto.peak_ == pytest.approx(2, rel=0.1)
    assert np.all(np.diff(kuramoto.mean_[couplings >= 2.2]) > 0)
