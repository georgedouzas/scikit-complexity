"""Double pendulum, a bob of mass `m2` on a rod of length `l2` hanging from a bob of mass `m1` on a rod `l1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


def _double_bobs(theta1: float, theta2: float, l1: float, l2: float, **_: float) -> np.ndarray:
    """Return the positions of the two bobs of a double pendulum hanging from the origin."""
    first = np.array([l1 * np.sin(theta1), -l1 * np.cos(theta1)])
    return np.stack([first, first + np.array([l2 * np.sin(theta2), -l2 * np.cos(theta2)])])


class DoublePendulumElement(Element):
    """Double pendulum, a bob of mass `m2` on a rod of length `l2` hanging from a bob of mass `m1` on a rod `l1`.

    The angles `theta1` and `theta2` of the rods from the vertical have order 2. The motion comes from
    `DoublePendulumInteraction`.

    Args:
        label:
            The label of the double pendulum.

        visual:
            How the double pendulum is drawn, or `None` for its class visual.

        theta1:
            The initial angle of the upper rod.

        theta1_d1:
            The initial angular velocity of the upper rod.

        theta2:
            The initial angle of the lower rod.

        theta2_d1:
            The initial angular velocity of the lower rod.

        m1:
            The mass of the upper bob.

        m2:
            The mass of the lower bob.

        l1:
            The length of the upper rod.

        l2:
            The length of the lower rod.

        g:
            The gravitational acceleration.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import DoublePendulumElement
        >>> DoublePendulumElement('pendulum', m2=2.0).m2
        2.0
    """

    visual = VisualSpec('bodies', {'positions': _double_bobs, 'pivot': (0.0, 0.0), 'rods': True, 'traced': [1]})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta1: Annotated[float, StateSpec(period=2 * pi, order=2)] = math.pi / 2,
        theta1_d1: float = 0.0,
        theta2: Annotated[float, StateSpec(period=2 * pi, order=2)] = math.pi / 2,
        theta2_d1: float = 0.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta1 = theta1
        self.theta1_d1 = theta1_d1
        self.theta2 = theta2
        self.theta2_d1 = theta2_d1
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
