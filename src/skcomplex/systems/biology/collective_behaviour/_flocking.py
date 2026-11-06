"""Define flocking, which steers every boid by cohesion, separation, and alignment with the boids around it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.components.interactions import CouplingInteraction
from ....core.engine.solvers import SystemView
from ....core.specs import ClampSpec, RadiusSpec, RoleSpec
from ....core.types import Array
from ._boid_view import BoidView


class FlockingInteraction(CouplingInteraction):
    """Reynolds' steering of every boid toward, away from, and along the boids it sees, with a bounded speed.

    The velocity of a boid gains the mean offset to the boids within `radius` weighted by `cohesion`, loses the summed
    offset to the boids within `separation_radius` weighted by `separation`, and gains the mean difference of their
    velocities from its own weighted by `alignment`. Its speed is then clipped between `min_speed` and `max_speed`.
    Offsets are the shortest ones on the torus of side `width` by `height`. It is the agent form of coupling over two
    radii of one priority, with a clamp of the norm.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of boids.

        phase:
            The phase of the step the steering runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import FlockingInteraction
        >>> [spec.reduce for spec in FlockingInteraction('flocking', target='flock').radii]
        ['mean', 'sum', 'mean']
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            write=('vx', 'vy'),
            form='agent',
            radii=(
                RadiusSpec('radius', self._get_cohesion, reduce='mean'),
                RadiusSpec('separation_radius', self._get_separation),
                RadiusSpec('radius', self._get_alignment, reduce='mean'),
            ),
            clamp=ClampSpec(min_norm='min_speed', max_norm='max_speed'),
            phase=phase,
        )

    @staticmethod
    def _get_offset(agent: BoidView, other: BoidView) -> Array:
        """Return the shortest offset from a boid to another on the torus."""
        size = np.array([agent.width, agent.height])
        offset = np.asarray(other.location, dtype=float) - np.asarray(agent.location, dtype=float)
        return (offset + size / 2) % size - size / 2

    def _get_cohesion(self, agent: BoidView, other: BoidView, system: SystemView) -> Array:
        """Return the pull of a boid toward another."""
        return agent.cohesion * self._get_offset(agent, other)

    def _get_separation(self, agent: BoidView, other: BoidView, system: SystemView) -> Array:
        """Return the push of a boid away from a close one."""
        return -agent.separation * self._get_offset(agent, other)

    @staticmethod
    def _get_alignment(agent: BoidView, other: BoidView, system: SystemView) -> Array:
        """Return the turn of a boid toward the velocity of another."""
        return agent.alignment * np.array([other.vx - agent.vx, other.vy - agent.vy])
