"""Define place visits, which count the steps a resident stands in its favourite place."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, cast

from ....core.interactions import MapInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ._resident_view import ResidentView


class PlaceVisitsInteraction(MapInteraction):
    """Count of the steps every resident stands in its favourite place, read from the regions of the layer `places`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of residents.

    Examples:
        >>> from skcomplex.models.social.organizations import PlaceVisitsInteraction
        >>> PlaceVisitsInteraction('visits', target='residents').mode
        'increment'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='visits',
            function=self._count,
            mode='increment',
            schedule='member',
            form='agent',
        )

    @staticmethod
    def _count(agent: AgentViewProtocol, model: ModelViewProtocol) -> float:
        """Return 1 for a resident that stands in its favourite place and 0 for any other."""
        resident = cast(ResidentView, agent)
        return float(int(resident.favourite) in resident.regions_here('places'))
