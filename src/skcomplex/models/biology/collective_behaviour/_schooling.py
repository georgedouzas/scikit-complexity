"""Define schooling, which turns every fish of a Couzin shoal by zones of repulsion, attraction, and alignment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated, cast

import numpy as np

from ....core.interactions import CouplingInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....engine.solvers import ModelView
from ....specs import ClampSpec, RadiusSpec, RoleSpec
from ....types import Array
from ._shoal_member_view import ShoalMemberView


class SchoolingInteraction(CouplingInteraction):
    """Couzin's turn of every fish toward a desired direction set by zones around it, by at most `max_turn`.

    When neighbours lie within the zone of repulsion `alpha`, the desired direction points away from them, and
    repulsion overrides the other zones. Otherwise it points toward the neighbours within the zone of interaction `rho`
    and along their headings and the fish's own, and an informed fish adds its preferred direction with the weight
    `omega`. A fish with no neighbour keeps its heading. Offsets are the shortest ones on the torus of side `size`. It
    is the agent form of coupling over two radii of different priorities, with a clamp of the turn.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of fish.

        phase:
            The phase of the step the turn runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.biology.collective_behaviour import SchoolingInteraction
        >>> [spec.priority for spec in SchoolingInteraction('schooling', target='shoal').radii]
        [1, 0, 0]
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            write=('vx', 'vy'),
            form='agent',
            combine=self._get_desired,
            radii=(
                RadiusSpec('alpha', self._get_repulsion, priority=1),
                RadiusSpec('rho', self._get_attraction),
                RadiusSpec('rho', self._get_alignment),
            ),
            clamp=ClampSpec(max_angle='max_turn'),
            phase=phase,
        )

    @staticmethod
    def _unit(vector: Array) -> Array:
        """Return a vector scaled to unit length, or the vector itself when it is zero."""
        norm = float(np.hypot(*vector))
        return vector / norm if norm > 0 else vector

    def _get_direction(self, agent: ShoalMemberView, other: ShoalMemberView) -> Array:
        """Return the unit direction from a fish to another along the shortest offset on the torus."""
        size = agent.size
        offset = np.asarray(other.location, dtype=float) - np.asarray(agent.location, dtype=float)
        return self._unit((offset + size / 2) % size - size / 2)

    def _get_repulsion(self, agent: ShoalMemberView, other: ShoalMemberView, model: ModelView) -> Array:
        """Return the direction away from a close neighbour."""
        return -self._get_direction(agent, other)

    def _get_attraction(self, agent: ShoalMemberView, other: ShoalMemberView, model: ModelView) -> Array:
        """Return the direction toward a neighbour."""
        return self._get_direction(agent, other)

    @staticmethod
    def _get_alignment(agent: ShoalMemberView, other: ShoalMemberView, model: ModelView) -> Array:
        """Return the heading of a neighbour."""
        return np.array([other.vx, other.vy])

    def _get_desired(
        self,
        agent: AgentViewProtocol,
        model: ModelViewProtocol,
        reduction: Array | None,
        priority: int | None,
    ) -> Array:
        """Return the desired direction from repulsion, or from attraction, alignment, and the preferred direction."""
        fish = cast(ShoalMemberView, agent)
        heading = np.array([fish.vx, fish.vy])
        if reduction is None:
            return heading
        if priority == 1:
            desired = self._unit(reduction)
        else:
            goal = np.array([math.cos(fish.preferred), math.sin(fish.preferred)])
            desired = self._unit(self._unit(heading + reduction) + fish.omega * goal)
        return desired if float(np.hypot(*desired)) > 0 else heading
