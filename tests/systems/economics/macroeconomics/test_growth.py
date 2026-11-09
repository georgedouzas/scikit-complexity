"""Test the `_growth` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.macroeconomics import (
    EconomyElement,
    MultiplierAcceleratorSystem,
    SolowGrowthInteraction,
)


@pytest.mark.parametrize('cv', [0.6, 1.2])
def test_multiplier_accelerator_stability(cv):
    """Test that the cycle damps when `c v < 1` and grows when `c v > 1`."""
    income = simulate(MultiplierAcceleratorSystem(v=cv / 0.8), until=60).values('economy.Y')
    deviation = np.abs(income - 100.0)
    assert (deviation[-10:].max() > deviation[:10].max()) == (cv > 1)


def test_solow_reaches_steady_state():
    """Test that Solow capital approaches `(s / delta) ** (1 / (1 - alpha))`."""
    s, alpha, delta = 0.3, 0.4, 0.1
    system = System(
        elements=[EconomyElement('country', k=1.0)],
        interactions=[SolowGrowthInteraction('solow', target='country', s=s, alpha=alpha, delta=delta)],
    )
    capital = simulate(system, until=300, dt=10.0).values('country.k')[-1]
    assert capital == pytest.approx((s / delta) ** (1 / (1 - alpha)), rel=1e-3)
