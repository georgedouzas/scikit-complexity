"""Define flights in any number of dimensions, which move every boid along its velocity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, cast

import numpy as np
from sage.sets.real_set import RealSet

from ....core.interactions import MovementInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import ParameterSpec, RoleSpec
from ....types import Array
from ._spatial_boid_view import SpatialBoidView


class SpatialFlightInteraction(MovementInteraction):
    """Flight of every boid by its velocity in one step, with the boundary of its space acting on the new position.

    Every boid moves in its own activation, after the rules and member interactions listed before it.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of boids.

        probability:
            The probability that a boid moves in a step.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import SpatialFlightInteraction
        >>> SpatialFlightInteraction('flight', target='flock').schedule
        'member'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, step=self._fly, probability=probability, schedule='member')

    @staticmethod
    def _fly(agent: AgentViewProtocol, model: ModelViewProtocol) -> Array:
        """Return the position of a boid after one step along its velocity."""
        boid = cast(SpatialBoidView, agent)
        return np.asarray(boid.location, dtype=float) + np.asarray(boid.velocity, dtype=float)
