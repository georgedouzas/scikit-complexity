"""Test the `_critical_exponent` module of `skcomplex.models.physics.realizations`."""

import math

import numpy as np
import pytest
from scipy import optimize

from skcomplex.analysis import ScalingAnalysis
from skcomplex.models.physics.realizations import CriticalExponentAnalysis


def test_critical_exponent_equals_the_power_law_of_mean_field_magnetization():
    """Test that beta of the mean-field Ising model is 1/2 and equals the scaling analysis of kind power."""
    temperatures = np.linspace(0.95, 0.995, 10)
    magnetizations = [optimize.brentq(lambda m, t=t: m - math.tanh(m / t), 1e-6, 1.0) for t in temperatures]
    beta = CriticalExponentAnalysis(parameters=temperatures, critical=1.0).check().prepare(magnetizations).run()
    general = ScalingAnalysis('power', parameters=temperatures, critical=1.0).check().prepare(magnetizations).run()
    np.testing.assert_array_equal(beta.exponent_, general.exponent_)
    assert beta.exponent_[0] == pytest.approx(0.5, abs=0.05)
