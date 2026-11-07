"""Axisymmetric Brownian particle in three dimensions, whose diffusion tensor turns with its axis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

AXES = ('x', 'y', 'z')
"""The names of the coordinates of the position."""

DIRECTIONS = ('ux', 'uy', 'uz')
"""The names of the components of the unit vector along the axis."""


class AnisotropicParticleElement(Element):
    """Axisymmetric Brownian particle in three dimensions, whose diffusion tensor turns with its axis.

    The position `r = (x, y, z)` diffuses with `D_par` along the unit axis `u = (ux, uy, uz)` and with `D_perp` across
    it, `dr = (sqrt(2 D_par) u u + sqrt(2 D_perp) (I - u u)) dW`. The axis diffuses on the unit sphere,
    `du = -2 D_r u dt + sqrt(2 D_r) (I - u u) dW_r` in the Itô form, and a boundary rescales it to unit length after
    every step against the drift of the discrete steps. The spin `psi` about the axis diffuses with `D_spin`, and its
    noise is correlated with the displacement along the axis through the coupling `D_c` of a chiral body, so that
    `<dr_par dpsi> = 2 D_c dt`, which needs `D_c² <= D_par D_spin`.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        ux:
            The initial first component of the axis.

        uy:
            The initial second component of the axis.

        uz:
            The initial third component of the axis.

        psi:
            The initial spin angle about the axis.

        D_par:
            The translational diffusion coefficient along the axis.

        D_perp:
            The translational diffusion coefficient across the axis.

        D_r:
            The rotational diffusion coefficient of the axis.

        D_spin:
            The rotational diffusion coefficient about the axis.

        D_c:
            The coupling of the translation along the axis and the rotation about it.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import AnisotropicParticleElement
        >>> AnisotropicParticleElement('janus').uz
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        z: Annotated[float, StateSpec()] = 0.0,
        ux: Annotated[float, StateSpec()] = 0.0,
        uy: Annotated[float, StateSpec()] = 0.0,
        uz: Annotated[float, StateSpec()] = 1.0,
        psi: Annotated[float, StateSpec()] = 0.0,
        D_par: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D_perp: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        D_spin: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        D_c: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.psi = psi
        self.D_par = D_par
        self.D_perp = D_perp
        self.D_r = D_r
        self.D_spin = D_spin
        self.D_c = D_c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the translation and the rotation with their correlated noises, and the boundary of the axis.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        axis = [getattr(s, name) for name in DIRECTIONS]
        position = [getattr(s, name) for name in AXES]
        length = sqrt(sum(component**2 for component in axis))
        laws: list[BaseLaw] = []
        for i, (coordinate, component) in enumerate(zip(position, axis, strict=True)):
            laws.append(RateLaw(component, -2 * s.D_r * component))
            for j, other in enumerate(axis):
                projector = (1 if i == j else 0) - component * other
                laws.append(NoiseLaw(coordinate, sqrt(2 * s.D_perp) * projector, process=f'translation_{j}'))
                laws.append(NoiseLaw(coordinate, sqrt(2 * s.D_par) * component * other, process=f'translation_{j}'))
                laws.append(NoiseLaw(component, sqrt(2 * s.D_r) * projector, process=f'rotation_{j}'))
            laws.append(BoundaryLaw(component, component / length))
        coupling = s.D_c / sqrt(s.D_par)
        for j, other in enumerate(axis):
            laws.append(NoiseLaw(s.psi, sqrt(2) * coupling * other, process=f'translation_{j}'))
        laws.append(NoiseLaw(s.psi, sqrt(2 * (s.D_spin - coupling**2)), process='spin'))
        return laws
