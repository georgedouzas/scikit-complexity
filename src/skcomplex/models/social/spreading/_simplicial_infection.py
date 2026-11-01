"""Infection of a susceptible node by the other members of a link or a filled triangle, when all are infected."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import HyperedgeInteraction
from ....core.laws import BaseLaw, JumpLaw
from ....core.laws.utils import where
from ....specs import ParameterSpec, RoleSpec

PAIR = 2


class SimplicialInfectionInteraction(HyperedgeInteraction):
    """Infection of a susceptible node by the other members of a link or a filled triangle, when all are infected.

    A susceptible node of a link is infected at rate `beta` when its neighbour is infected, and a susceptible node of a
    triangle at rate `beta_triangle` when both other members are infected.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of nodes.

        beta:
            The rate of infection through a link.

        beta_triangle:
            The rate of infection through a triangle.

    Examples:
        >>> from skcomplex.models.social.spreading import SimplicialInfectionInteraction
        >>> SimplicialInfectionInteraction('infection', target='nodes', beta_triangle=0.5).beta_triangle
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        beta_triangle: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.beta = beta
        self.beta_triangle = beta_triangle

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection of the focal node when every other member of its link or triangle is infected."""
        infected = s.member.infected
        rate = s.beta if len(s.others) + 1 == PAIR else s.beta_triangle
        exposed = math.prod((other.infected for other in s.others), start=where(infected == 0, 1, 0))
        return [JumpLaw(rate * exposed, {infected: 1})]
