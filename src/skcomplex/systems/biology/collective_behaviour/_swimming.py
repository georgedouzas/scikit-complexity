"""Define swimming, which moves every fish of a shoal along its heading after an angular noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated, cast

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.interactions import MovementInteraction
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.specs import ParameterSpec, RoleSpec
from ._shoal_member_view import ShoalMemberView


class SwimmingInteraction(MovementInteraction):
    """Swim of every fish by `speed` along its heading, turned first by its angular noise `deviation`.

    The heading `(vx, vy)` stays a unit vector, and the fish moves on the torus of side `size`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of fish.

        probability:
            The probability that a fish swims in a step.

        phase:
            The phase of the step the swim runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import SwimmingInteraction
        >>> SwimmingInteraction('swimming', target='shoal', phase='swim').phase
        'swim'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        probability: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            step=self._swim,
            probability=probability,
            schedule='member',
            phase=phase,
        )

    @staticmethod
    def _swim(agent: AgentViewProtocol, system: SystemViewProtocol) -> tuple[float, ...]:
        """Turn the heading of a fish by its angular noise and return its position after one swim."""
        fish = cast(ShoalMemberView, agent)
        angle = math.atan2(fish.vy, fish.vx) + fish.deviation
        fish.vx, fish.vy = math.cos(angle), math.sin(angle)
        position = np.asarray(fish.location, dtype=float) + fish.speed * np.array([fish.vx, fish.vy])
        return tuple(float(item) for item in np.mod(position, fish.size))
