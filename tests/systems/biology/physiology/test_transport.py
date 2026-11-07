"""Test the `_transport` module of `skcomplex.systems.biology.physiology`."""

import numpy as np

from skcomplex.systems.biology.physiology import AdaptiveFlowNetworkSystem

from ..conftest import _run_until


def test_vessels_settle_at_the_pressure_drop_of_balanced_adaptation():
    """Test that every vessel ends with the pressure drop `(alpha2 / alpha1)^(1 / gamma)`, the fixed point of Eq.

    (72).
    """
    system = AdaptiveFlowNetworkSystem(alpha1=1.0, alpha2=4.0, gamma=2.0).prepare().system_
    conductances = _run_until(system, 40, dt=1.0).values('vessels.C')[-1]
    drops = np.abs(system.interactions[0]._pressure_drops(conductances))
    np.testing.assert_allclose(drops, 2.0, rtol=1e-3)


def test_vessels_near_the_source_thicken():
    """Test that the vessels at the source carry the largest conductances."""
    conductances = _run_until(AdaptiveFlowNetworkSystem(), 40, dt=1.0).values('vessels.C')[-1]
    assert conductances[:2].min() > np.median(conductances)
