"""Define heading alignment, which turns every particle to the mean heading of its neighbours plus noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, cos

from ....core.interactions import CouplingInteraction
from ....core.laws.utils import uniform, where
from ....core.neighborhoods import RadiusNeighborhood
from ....core.types import Target
from ....specs import RoleSpec


class HeadingAlignmentInteraction(CouplingInteraction):
    """Vicsek alignment, the turn of every particle to the circular mean heading of its neighbours plus noise.

    Every step the heading `theta` of every particle becomes the circular mean of the headings within the unit
    distance, itself included, plus a uniform angle in `[-eta pi, eta pi]` with the parameter `eta` of the particle.
    With `nematic`, every heading is flipped to the side nearer the particle's own heading first, as for particles
    with an axis rather than a direction. It is the map form of coupling with the circular mean over a radius.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of particles.

        nematic:
            Whether every heading is flipped to the side nearer the particle's own heading.

    Examples:
        >>> from skcomplex.models.physics.active_matter import HeadingAlignmentInteraction
        >>> HeadingAlignmentInteraction('alignment', target='flock', nematic=True).reduce
        'circular-mean'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], nematic: bool = False) -> None:
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
            neighborhood=RadiusNeighborhood(1.0, center=True),
        )
        self.nematic = nematic

    @staticmethod
    def _get_heading(namespace: SimpleNamespace) -> Expression:
        """Return the heading of the neighbour."""
        return namespace.other.theta

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
