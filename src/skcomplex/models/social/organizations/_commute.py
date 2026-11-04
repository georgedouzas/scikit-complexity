"""Define commutes, which move residents between their homes, their work, and their favourite places."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, cast

import numpy as np

from ....core.interactions import MovementInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ._resident_view import ResidentView

AT_WORK = 1
AT_PLACE = 2
STOPS = 3


class CommuteInteraction(MovementInteraction):
    """Commute of every resident in a cycle of three steps: to work, to its favourite place, and back home.

    The work and the favourite place are regions of the layers `facilities` and `places`, so a resident stands at a new
    uniform point of the region at every visit, and at the same point of its home.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of residents.

    Examples:
        >>> from skcomplex.models.social.organizations import CommuteInteraction
        >>> CommuteInteraction('commute', target='residents').schedule
        'member'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, step=self._commute, schedule='member')

    @staticmethod
    def _commute(agent: AgentViewProtocol, model: ModelViewProtocol) -> object:
        """Return the destination of a resident at the current step, its work, its favourite place, or its home."""
        resident = cast(ResidentView, agent)
        if np.isnan(resident.home).any():
            resident.home = np.asarray(resident.location, dtype=float)
        stop = int(model.time) % STOPS
        if stop == AT_WORK:
            return model.region('facilities', int(resident.work))
        if stop == AT_PLACE:
            return model.region('places', int(resident.favourite))
        return resident.home
