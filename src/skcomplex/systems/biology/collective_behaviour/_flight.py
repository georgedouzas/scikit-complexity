"""Define flight, which moves every boid along its velocity on a torus."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, cast

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.interactions import MovementInteraction
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.specs import ParameterSpec, RoleSpec
from ._boid_view import BoidView


class FlightInteraction(MovementInteraction):
    """Flight of every boid by its velocity `(vx, vy)` in one step, on the torus of side `width` by `height`.

    Every boid moves in its own activation, after the rules and member interactions listed before it.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of boids.

        probability:
            The probability that a boid moves in a step.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import FlightInteraction
        >>> FlightInteraction('flight', target='flock').schedule
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
    def _fly(agent: AgentViewProtocol, system: SystemViewProtocol) -> tuple[float, ...]:
        """Return the position of a boid after one step along its velocity."""
        boid = cast(BoidView, agent)
        position = np.asarray(boid.location, dtype=float) + np.array([boid.vx, boid.vy])
        return tuple(float(item) for item in np.mod(position, np.array([boid.width, boid.height])))
