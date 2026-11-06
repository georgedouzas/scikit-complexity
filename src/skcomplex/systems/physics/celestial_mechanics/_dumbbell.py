"""Rigid dumbbell orbiting a primary: the position of its centre of mass and the orientation of its arm."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class DumbbellElement(Element):
    """Rigid dumbbell orbiting a primary: the position of its centre of mass and the orientation of its arm.

    The cylindrical coordinates `(u, theta, v)` place the centre of mass of the dumbbell relative to the primary, and
    the spherical angles `(phi, psi)` give the direction of its arm, of length `epsilon` in units of the mean orbit
    radius. The mass `1 - delta` of the dumbbell sits at `delta epsilon` along the arm and the mass `delta` at
    `1 - delta` against it. All five coordinates are states of order 2 that a Lagrangian moves.

    Args:
        label:
            The label of the dumbbell.

        visual:
            How the dumbbell is drawn, or `None` for `default_visual_`.

        u:
            The initial distance from the axis through the primary.

        u_d1:
            The initial radial velocity.

        theta:
            The initial orbital angle.

        theta_d1:
            The initial orbital angular velocity.

        v:
            The initial height above the orbital plane.

        v_d1:
            The initial vertical velocity.

        phi:
            The initial azimuth of the arm.

        phi_d1:
            The initial spin of the arm about the vertical.

        psi:
            The initial polar angle of the arm, `pi / 2` in the orbital plane.

        psi_d1:
            The initial rate of the polar angle of the arm.

        delta:
            The share of the second mass in the mass of the dumbbell, `1 / 2` for equal masses.

        epsilon:
            The length of the arm in units of the mean orbit radius.

        mu:
            The share of the primary in the mass of the three bodies.

    Attributes:
        default_visual_:
            The visual of the dumbbell when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import DumbbellElement
        >>> DumbbellElement('dumbbell').delta
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec(order=2)] = 1.0,
        u_d1: float = 0.0,
        theta: Annotated[float, StateSpec(order=2)] = 0.0,
        theta_d1: float = 1.0,
        v: Annotated[float, StateSpec(order=2)] = 0.0,
        v_d1: float = 0.0,
        phi: Annotated[float, StateSpec(order=2)] = 0.0,
        phi_d1: float = 1.0,
        psi: Annotated[float, StateSpec(order=2)] = math.pi / 2,
        psi_d1: float = 0.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.5,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.u_d1 = u_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.v = v
        self.v_d1 = v_d1
        self.phi = phi
        self.phi_d1 = phi_d1
        self.psi = psi
        self.psi_d1 = psi_d1
        self.delta = delta
        self.epsilon = epsilon
        self.mu = mu
