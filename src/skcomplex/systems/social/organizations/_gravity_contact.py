"""The interaction of a resident with the others, which decays as a power of the distance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.base import BasePart
from ....core.components.interactions import Interaction
from ....core.components.laws import EquationLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import RadiusNeighborhood
from ....core.specs import ParameterSpec, RoleSpec


class GravityContactInteraction(Interaction):
    """Contacts of every resident, the sum of `r^-gamma` over the others within a reach.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of residents.

        reach:
            The largest distance of a contact, which the diameter of a city covers.

        gamma:
            The decay exponent of an interaction with the distance.

    Examples:
        >>> from skcomplex.systems.social.organizations import GravityContactInteraction
        >>> GravityContactInteraction('contact', target='residents', reach=30.0, gamma=1.5).gamma
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        reach: float,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.reach = reach
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the equation of the contacts of a resident.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation of the contacts.
        """
        dx, dy = s.target.other.x[0] - s.target.x[0], s.target.other.x[1] - s.target.x[1]
        contacts = neighbors((dx**2 + dy**2) ** (-s.gamma / 2), over=RadiusNeighborhood(self.reach))
        return [EquationLaw(s.target.contacts == contacts, target=s.target.contacts)]
