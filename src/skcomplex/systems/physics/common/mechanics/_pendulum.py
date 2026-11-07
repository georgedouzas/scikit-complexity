"""Pendulum, a bob of mass `m` on a rigid rod of length `l` that swings by the angle `theta` under gravity `g`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


def _pendulum_bob(theta: float, l: float, **_: float) -> list[list[float]]:
    """Return the position of the bob of a pendulum hanging from the origin."""
    return [[l * np.sin(theta), -l * np.cos(theta)]]


class PendulumElement(Element):
    """Pendulum, a bob of mass `m` on a rigid rod of length `l` that swings by the angle `theta` under gravity `g`.

    The angle has order 2 and its first derivative `theta_d1` is the angular velocity. The motion comes from an
    interaction such as `PendulumInteraction`.

    Args:
        label:
            The label of the pendulum.

        visual:
            How the pendulum is drawn, or `None` for its class visual.

        theta:
            The initial angle from the vertical, in radians.

        theta_d1:
            The initial angular velocity.

        m:
            The mass of the bob.

        g:
            The gravitational acceleration.

        l:
            The length of the rod.

    Examples:
        >>> from skcomplex.systems.physics.common.mechanics import PendulumElement
        >>> PendulumElement('bob', theta=0.2, l=2.0).l
        2.0
    """

    visual = VisualSpec('bodies', {'positions': _pendulum_bob, 'pivot': (0.0, 0.0), 'rods': True})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.5,
        theta_d1: float = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.g = g
        self.l = l
