"""Spring that ties a particle to an anchor point, which oscillates transversally as `amplitude sin(omega t)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, oo, sin, vector
from sage.sets.real_set import RealSet

from ....specs import ParameterSpec, RoleSpec
from ..common.particles import HarmonicPotentialInteraction


class DrivenAnchorInteraction(HarmonicPotentialInteraction):
    """Spring that ties a particle to an anchor point, which oscillates transversally as `amplitude sin(omega t)`.

    The energy is `k |x - center - (0, amplitude sin(omega t))|² / 2`, and a zero amplitude gives a fixed anchor.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The stiffness of the spring.

        center:
            The rest position of the anchor.

        amplitude:
            The amplitude of the transverse motion of the anchor.

        omega:
            The angular frequency of the anchor.

    Examples:
        >>> from skcomplex.models.physics.waves import DrivenAnchorInteraction
        >>> DrivenAnchorInteraction('end', target='mass_0', center=[0.0, 0.0], amplitude=0.1).amplitude
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        amplitude: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, k=k, center=center)
        self.amplitude = amplitude
        self.omega = omega

    def _get_center(self, s: SimpleNamespace) -> Expression:
        """Return the position `center + (0, amplitude sin(omega t))` of the anchor."""
        return s.center + vector(SR, [0, s.amplitude * sin(s.omega * s.t)])
