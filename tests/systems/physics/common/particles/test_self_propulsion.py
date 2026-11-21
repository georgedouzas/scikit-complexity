"""Test the `_self_propulsion` module of `skcomplex.systems.physics.common.particles`."""

import math
from types import SimpleNamespace

import numpy as np

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.physics.common.particles import ActiveParticleElement, SelfPropulsionInteraction


class TurningInteraction(Interaction):
    """Turning of a heading at the angular velocity `omega`."""

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, omega: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.omega = omega

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the turning of the heading of the target."""
        super()._prepare(context)
        self.laws_ = [RateLaw(context.target.theta.diff(context.t), context.omega)]


def test_position_drifts_along_a_turning_heading():
    """Test that a position driven along a heading turning at `omega` traces a circle of radius `speed / omega`."""
    swimmer = ActiveParticleElement('swimmer', x=[0.0, 0.0], theta=0.0)
    propulsion = SelfPropulsionInteraction('propulsion', target='swimmer', speed=2.0)
    turning = TurningInteraction('turning', target='swimmer', omega=1.0)
    system = System(elements=[swimmer], interactions=[propulsion, turning], space=ContinuousSpace(n_dim=2)).prepare()
    trajectory = system.simulate(until=math.pi, dt=0.1)
    np.testing.assert_allclose(trajectory.get_values('swimmer.x')[-1], [0.0, 4.0], atol=1e-5)
