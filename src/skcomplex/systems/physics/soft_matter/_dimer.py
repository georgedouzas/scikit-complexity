"""Two Brownian or active spheres joined by a rigid rod, described by the centre and the angle of the rod."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, cos, oo, pi, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class DimerElement(Element):
    """Two Brownian or active spheres joined by a rigid rod, described by the centre and the angle of the rod.

    Each sphere diffuses with `D_t` and swims at the speed `v0` along its own orientation, `theta1` or `theta2`, which
    diffuse with `D_r`, and the rod keeps them at the distance `L`. Moving both spheres freely and then restoring
    their distance moves the centre `c` by the mean of their displacements and turns the rod by the part of their
    relative displacement across it, so in the limit of small steps `dc = v0 (n1 + n2) / 2 dt + sqrt(D_t) dW` and
    `dphi = v0 (n1 - n2) . e / L dt + 2 sqrt(D_t) / L dW`, with `e = (-sin phi, cos phi)` across the rod.

    Args:
        label:
            The label of the dimer or the group.

        visual:
            How the dimer is drawn, or `None` for `default_visual_`.

        c:
            The initial centre.

        phi:
            The initial angle of the rod.

        theta1:
            The initial orientation of the first sphere.

        theta2:
            The initial orientation of the second sphere.

        L:
            The length of the rod.

        v0:
            The swimming speed of each sphere, 0 for Brownian spheres.

        D_t:
            The translational diffusion coefficient of each sphere.

        D_r:
            The rotational diffusion coefficient of the orientation of each sphere.

    Attributes:
        default_visual_:
            The visual of the dimer when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import DimerElement
        >>> DimerElement('molecule').L
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        phi: Annotated[float, StateSpec(role='direction')] = 0.0,
        theta1: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        theta2: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.phi = phi
        self.theta1 = theta1
        self.theta2 = theta2
        self.L = L
        self.v0 = v0
        self.D_t = D_t
        self.D_r = D_r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion of the centre and the turning of the rod, and the diffusion of the orientations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first = vector(SR, [cos(s.theta1), sin(s.theta1)])
        second = vector(SR, [cos(s.theta2), sin(s.theta2)])
        across = vector(SR, [-sin(s.phi), cos(s.phi)])
        return [
            RateLaw(s.c, s.v0 * (first + second) / 2),
            NoiseLaw(s.c, sqrt(s.D_t)),
            RateLaw(s.phi, s.v0 * (first - second).dot_product(across) / s.L),
            NoiseLaw(s.phi, 2 * sqrt(s.D_t) / s.L),
            NoiseLaw(s.theta1, sqrt(2 * s.D_r)),
            NoiseLaw(s.theta2, sqrt(2 * s.D_r)),
        ]
