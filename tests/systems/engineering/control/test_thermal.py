"""Test the `_thermal` module of `skcomplex.systems.engineering.control`."""

import numpy as np

from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ConnectorSpec
from skcomplex.systems.engineering.control import (
    THERMAL,
    ConvectionElement,
    HeatCapacityElement,
    ThermalConductorElement,
)


def test_thermal_connector_is_a_connector():
    """Test that the thermal connector carries a temperature and a heat flow."""
    assert isinstance(THERMAL, ConnectorSpec)
    assert (THERMAL.potential, THERMAL.flow) == ('T', 'Q')


def test_body_cools_through_a_conductor_and_convection_in_closed_form():
    """Test that a body cools towards a fluid at the rate of the conductance in series with the convection."""
    system = System(
        elements=[
            HeatCapacityElement('body', T=80.0, C=4.0),
            ThermalConductorElement('wall', G=2.0),
            ConvectionElement('surface', h=2.0, fluid=20.0),
        ],
        interactions=[
            ConnectionInteraction('inner', members=['body', 'wall'], ports=['a', 'a']),
            ConnectionInteraction('outer', members=['wall', 'surface'], ports=['b', 'a']),
        ],
    )
    trajectory = Simulation(system, dt=0.5).prepare().run(until=5.0).trajectory_
    expected = 20.0 + 60.0 * np.exp(-trajectory.times * (2.0 * 2.0 / (2.0 + 2.0)) / 4.0)
    assert np.allclose(trajectory.values('body.T').ravel(), expected, rtol=1e-4)
    assert np.allclose(trajectory.values('wall.b_T').ravel(), (expected + 20.0) / 2, rtol=1e-4)
