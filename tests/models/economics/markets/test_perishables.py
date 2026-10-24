"""Test the `_perishables` module of `skcomplex.models.economics.markets`."""

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.economics.markets import perishable_goods_duopoly


def _asymmetry_at_mean_field(temperature, control):
    """Return the stable asymmetry of the mean-field equation (4.48) of the perishable goods model."""

    def drift(m):
        return temperature * np.log((1 - m) / (1 + m)) + 1 / (1 + control / (1 + m)) - 1 / (1 + control / (1 - m))

    return brentq(drift, 1e-6, 1 - 1e-12)


@pytest.mark.parametrize('ratio', [0.5, 2.0])
def test_perishable_goods_duopoly_symmetry_breaking(ratio):
    """Test that one shop takes most customers below T_c = R / (1 + R)^2 and that both shops share them above it."""
    control = 1.0
    critical = control / (1 + control) ** 2
    system = perishable_goods_duopoly(200, stock=0.5, tau_a=1.0, T=ratio * critical, random_state=0)
    asymmetry = np.abs(
        Simulation(system, random_state=0, observables=describe('perishable_goods_duopoly').observables_of(system))
        .run(until=300)
        .trajectory_['m'][-50:],
    ).mean()
    if ratio < 1:
        assert asymmetry == pytest.approx(_asymmetry_at_mean_field(ratio * critical, control), abs=0.1)
    else:
        assert asymmetry < 0.05
