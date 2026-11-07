"""Define heading alignment, which turns every particle to the mean heading of its neighbours plus noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, atan2, cos, sin

from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import lag, uniform, where
from ....core.components.spaces import BaseNeighborhood, RadiusNeighborhood
from ....core.components.types import Target
from ....core.specs import RoleSpec


class HeadingAlignmentInteraction(CouplingInteraction):
    """Vicsek alignment, the turn of every particle to the circular mean heading of its neighbours plus noise.

    Every step the heading `theta` of every particle becomes the circular mean of the headings within the unit
    distance, itself included, plus a uniform angle in `[-eta pi, eta pi]` with the parameter `eta` of the particle.
    With `nematic`, every heading is flipped to the side nearer the particle's own heading first, as for particles
    with an axis rather than a direction. It is the map form of coupling with the circular mean over a radius. Another
    neighbourhood, such as the `k` nearest particles or a vision cone, replaces the unit disk.

    With a positive `latency` `h`, every particle reads the headings of its neighbours `h` steps ago. With a negative
    latency it reads their headings predicted `|h|` steps ahead, by a least-squares line through their last `memory`
    headings, unwrapped by taking every change between two steps as the smallest angle. A latent alignment needs
    particles that keep enough past headings, such as `DelayedVicsekParticleElement`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of particles.

        nematic:
            Whether every heading is flipped to the side nearer the particle's own heading.

        neighborhood:
            The neighbourhood of every particle, itself included, or `None` for the unit disk.

        latency:
            The number of steps by which the headings of the neighbours are read late, or predicted ahead when
            negative.

        memory:
            The number of past headings a prediction fits, at least 2.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import HeadingAlignmentInteraction
        >>> HeadingAlignmentInteraction('alignment', target='flock', nematic=True).reduce
        'circular-mean'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        nematic: bool = False,
        neighborhood: BaseNeighborhood | None = None,
        latency: int = 0,
        memory: int = 5,
    ) -> None:
        super().__init__(
            label,
            target=target,
            read='theta',
            write='theta',
            function=self._get_heading,
            reduce='circular-mean',
            weights=self._get_side,
            form='map',
            combine=self._get_noisy,
            neighborhood=RadiusNeighborhood(1.0, center=True) if neighborhood is None else neighborhood,
        )
        self.nematic = nematic
        self.latency = latency
        self.memory = memory

    def _get_heading(self, namespace: SimpleNamespace) -> Expression:
        """Return the heading of the neighbour, delayed or predicted."""
        heading = namespace.other.theta
        if self.latency >= 0:
            return lag(heading, self.latency)
        unwrapped = [heading]
        for step in range(1, self.memory):
            change = lag(heading, step - 1) - lag(heading, step)
            unwrapped.append(unwrapped[-1] - atan2(sin(change), cos(change)))
        times = [-step for step in range(self.memory)]
        mean_time = sum(times) / self.memory
        mean_heading = sum(unwrapped) / self.memory
        spread = sum((time - mean_time) ** 2 for time in times)
        slope = sum((time - mean_time) * (value - mean_heading) for time, value in zip(times, unwrapped, strict=True))
        return mean_heading + slope / spread * (-self.latency - mean_time)

    def _get_side(self, namespace: SimpleNamespace) -> Expression:
        """Return -1 for a neighbour heading away from the particle when nematic, and 1 otherwise."""
        if not self.nematic:
            return SR(1)
        return where(cos(namespace.other.theta - namespace.member.theta) >= 0, 1, -1)

    @staticmethod
    def _get_noisy(namespace: SimpleNamespace, reduction: Target) -> Target:
        """Return the mean heading plus a uniform angle in `[-eta pi, eta pi]`."""
        eta = namespace.member.eta
        return reduction + uniform('noise', -eta * math.pi, eta * math.pi)
