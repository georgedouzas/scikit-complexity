"""Tilted periodic potential `U(x) - f x`, with `U(x) = U0 cos(2 pi x / L)`, on overdamped particles on a line."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import GradientInteraction
from ....core.specs import ParameterSpec, RoleSpec


class TiltedWashboardInteraction(GradientInteraction):
    """Tilted periodic potential `U(x) - f x`, with `U(x) = U0 cos(2 pi x / L)`, on overdamped particles on a line.

    The force divides by the friction `gamma` of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or the group.

        U0:
            The amplitude of the periodic potential, half its barrier height.

        L:
            The period of the potential.

        f:
            The constant non-conservative force that tilts the potential.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TiltedWashboardInteraction
        >>> TiltedWashboardInteraction('washboard', target='beads', f=0.5).f
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        U0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        f: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, target=target, states='x', divisor='gamma')
        self.U0 = U0
        self.L = L
        self.f = f

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return the tilted potential `U0 cos(2 pi x / L) - f x`.

        Args:
            s:
                The namespace of the symbols the potential can use.

        Returns:
            potential:
                The potential.
        """
        position = s.target.x[0]
        return s.U0 * cos(2 * math.pi * position / s.L) - s.f * position
