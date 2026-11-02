"""Encounters that slow pedestrians down in crowded rooms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import EncounterInteraction
from ....specs import ParameterSpec, RoleSpec


def crowding_table(rooms: int, speeds: int, step: int) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the rate pattern and the transition table of encounters that change the speed class by one step.

    With `step = -1`, two pedestrians in the same room meet and the candidate slows down one class, so crowded rooms
    walk slower. With `step = 1`, a pedestrian meets the exit signs and speeds up one class.

    Args:
        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

        step:
            `-1` to slow down in encounters within a room, `1` to speed up in encounters with the exit signs.

    Returns:
        pattern:
            The rate pattern, 1 for the encounters that act.

        table:
            The transition table.

    Examples:
        >>> from skcomplex.models.social.crowds import crowding_table
        >>> crowding_table(1, 2, 1)[0].ravel().tolist()
        [1.0, 0.0, 0.0, 0.0]
    """
    sites = (rooms + 1) * speeds
    fields = sites if step < 0 else 1
    pattern = np.zeros((sites, fields))
    table = np.zeros((sites, fields, sites))
    for site in range(sites):
        room, speed = divmod(site, speeds)
        outcome = min(max(speed + step, 0), speeds - 1)
        table[site, :, room * speeds + outcome] = 1.0
        if room == rooms or outcome == speed:
            continue
        if step > 0:
            pattern[site, 0] = 1.0
        else:
            pattern[site, room * speeds : (room + 1) * speeds] = 1.0
    return pattern, table


class CrowdingInteraction(EncounterInteraction):
    """Encounters that slow pedestrians down in crowded rooms.

    It is an `EncounterInteraction` of the crowd with itself. Two pedestrians in the same room meet at the rate
    `crowding`, and the candidate slows down one speed class. Pedestrians outside keep their speed.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the crowd, a distribution over rooms and speed classes.

        field:
            The path of the crowd that the candidates meet, the crowd again.

        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

        crowding:
            The rate of the encounters within a room that slow pedestrians down.

    Examples:
        >>> from skcomplex.models.social.crowds import CrowdingInteraction
        >>> CrowdingInteraction('crowding', candidate='crowd', field='crowd').crowding
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        rooms: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 7,
        speeds: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
        crowding: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.rooms = rooms
        self.speeds = speeds
        self.crowding = crowding

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the rate of the slowing encounters of every pair of classes within a room."""
        return crowding_table(int(self.rooms), int(self.speeds), -1)[0] * s.crowding

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table that moves a pedestrian down one speed class."""
        return crowding_table(int(self.rooms), int(self.speeds), -1)[1]
