"""Magnet below the plane of a pendulum bob, which attracts it with energy `-gamma / sqrt(|x - center|**2 + d**2)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.specs import ParameterSpec, RoleSpec
from ..common.particles import (
    PotentialEnergyInteraction,
)


class MagnetInteraction(PotentialEnergyInteraction):
    """Magnet below the plane of a pendulum bob, which attracts it with energy `-gamma / sqrt(|x - center|**2 + d**2)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the bob.

        center:
            The position of the magnet in the plane.

        gamma:
            The strength of the magnet.

        d:
            The height of the bob above the plane of the magnets.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import MagnetInteraction
        >>> MagnetInteraction('magnet', target='bob', center=[1.0, 0.0]).gamma
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, target=target)
        self.center = center
        self.gamma = gamma
        self.d = d

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `-gamma / sqrt(|x - center|**2 + d**2)`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        displacement = s.target.x - s.center
        return -s.gamma / sqrt(displacement.dot_product(displacement) + s.d**2)
