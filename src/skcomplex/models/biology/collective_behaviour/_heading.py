"""Define headings, which steer the velocity of every boid by the boids around it, in any number of dimensions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, cast

import numpy as np

from ....core.interactions import MapInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ....types import Array
from ._spatial_boid_view import SpatialBoidView


class HeadingInteraction(MapInteraction):
    """Steering of the velocity of every boid by cohesion, separation, and alignment, with a bounded speed.

    The velocity of a boid gains the mean offset to the boids within `radius` weighted by `cohesion`, loses the summed
    offset to the boids within `separation_radius` weighted by `separation`, and gains the mean difference of their
    velocities from its own weighted by `alignment`. Its speed is then clipped between `min_speed` and `max_speed`.
    Offsets are the shortest ones of the space. It is the agent form of a map of the velocity under the `'member'`
    schedule.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of boids.

        phase:
            The phase of the step the steering runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import HeadingInteraction
        >>> HeadingInteraction('heading', target='flock').schedule
        'member'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            state='velocity',
            function=self._steer,
            schedule='member',
            form='agent',
            phase=phase,
        )

    @staticmethod
    def _get_steering(boid: SpatialBoidView, others: list[SpatialBoidView]) -> Array:
        """Return the change of the velocity of a boid from the boids it sees."""
        offsets = np.array([boid.displacement_to(other) for other in others])
        velocities = np.array([other.velocity for other in others])
        close = np.linalg.norm(offsets, axis=1) < boid.separation_radius
        return (
            boid.cohesion * offsets.mean(axis=0)
            - boid.separation * offsets[close].sum(axis=0)
            + boid.alignment * (velocities.mean(axis=0) - boid.velocity)
        )

    @staticmethod
    def _steer(agent: AgentViewProtocol, model: ModelViewProtocol) -> Array:
        """Return the velocity of a boid after one step of steering, with its speed clipped."""
        boid = cast(SpatialBoidView, agent)
        others = [cast(SpatialBoidView, other) for other in boid.neighbors(boid.radius)]
        velocity = np.asarray(boid.velocity, dtype=float)
        if others:
            velocity = velocity + HeadingInteraction._get_steering(boid, others)
        speed = float(np.linalg.norm(velocity))
        return velocity * np.clip(speed, boid.min_speed, boid.max_speed) / speed if speed else velocity
