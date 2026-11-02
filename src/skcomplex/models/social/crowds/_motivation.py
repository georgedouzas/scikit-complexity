"""Driving force of the social force model, which relaxes a walker's velocity to its desired velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import (
    RelaxationInteraction,
)
from ....specs import ParameterSpec, RoleSpec


class MotivationInteraction(RelaxationInteraction):
    """Driving force of the social force model, which relaxes a walker's velocity to its desired velocity.

    The force per unit mass is `(speed * direction - x_d1) / tau`, where the velocity `x_d1` is the first derivative of
    the position `x`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walker or group of walkers.

        speed:
            The desired speed.

        direction:
            The unit vector of the desired direction.

        tau:
            The relaxation time.

    Examples:
        >>> from skcomplex.models.social.crowds import MotivationInteraction
        >>> MotivationInteraction('drive', target='walkers', direction=[1.0, 0.0]).speed
        1.34
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.34,
        direction: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, target=target, state='x_d1')
        self.speed = speed
        self.direction = direction
        self.tau = tau

    def _get_goal(self, namespace: SimpleNamespace) -> object:
        """Return the desired velocity."""
        return namespace.speed * namespace.direction

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time."""
        return namespace.tau
