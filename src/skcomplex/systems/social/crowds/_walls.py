"""Repulsion of walkers by two walls of a corridor at `y = 0` and `y = width`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import (
    GradientInteraction,
)
from ....core.specs import ParameterSpec, RoleSpec


class WallsInteraction(GradientInteraction):
    """Repulsion of walkers by two walls of a corridor at `y = 0` and `y = width`.

    It is a `GradientInteraction` of the position `x` down the potential
    `strength falloff (exp((radius - y) / falloff) + exp((radius - width + y) / falloff))`, so the force per unit mass
    from a wall at distance `d` is `strength * exp((radius - d) / falloff)`, pointing away from the wall.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walker or group of walkers.

        width:
            The distance between the walls.

        strength:
            The strength of the repulsion.

        falloff:
            The distance over which the repulsion falls off.

        radius:
            The radius of a walker.

    Examples:
        >>> from skcomplex.systems.social.crowds import WallsInteraction
        >>> WallsInteraction('walls', target='walkers', width=5.0).width
        5.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        falloff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.2,
        radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, target=target, potential=self._get_potential)
        self.width = width
        self.strength = strength
        self.falloff = falloff
        self.radius = radius

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return the potential of both walls at the position of a walker."""
        y = s.target.x[1]
        return s.strength * s.falloff * (exp((s.radius - y) / s.falloff) + exp((s.radius - s.width + y) / s.falloff))
