"""Test the `_tipping_types` module of `skcomplex.systems.physics.climate`."""

import numpy as np

from skcomplex.systems import describe
from skcomplex.systems.physics.climate import (
    ramped_stochastic_fold,
    sutera_fraedrich_tipping,
)

from ..conftest import _run_default


def test_sutera_fraedrich_tipping_has_a_critical_rate():
    """Test that a slow ramp keeps the warm state and a fast one tips it, around the critical rate 0.175."""
    solver = describe('sutera_fraedrich_tipping').defaults['solver']
    slow = _run_default(sutera_fraedrich_tipping('rate', rho=0.16), until=300.0, dt=1.0, solver=solver).values(
        'earth.T',
    )
    fast = _run_default(sutera_fraedrich_tipping('rate', rho=0.2), until=300.0, dt=1.0, solver=solver).values('earth.T')
    assert slow.min() > 290
    assert slow[-1] > 300
    assert fast[-1] < 0


def test_ramped_stochastic_fold_tips_late_for_fast_ramps():
    """Test that the state tips after the control passes the static fold, later for faster ramps."""
    fold = 2 / (3 * np.sqrt(3))
    solver = describe('ramped_stochastic_fold').defaults['solver']
    controls = []
    for rate in (0.004, 0.02, 0.1):
        x = _run_default(ramped_stochastic_fold(rate=rate), until=2.0 / rate, dt=0.1, solver=solver).values('system.x')
        controls.append(-1 + rate * 0.1 * np.argmax(x > 0))
    assert fold < controls[0] < controls[1] < controls[2]
