"""Encounters with the exit signs that speed pedestrians up again."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import EncounterInteraction
from ....specs import ParameterSpec, RoleSpec
from ._crowding import crowding_table


class GuidanceInteraction(EncounterInteraction):
    """Encounters with the exit signs that speed pedestrians up again.

    It is an `EncounterInteraction` of the crowd with the exit signs. A pedestrian meets the exit signs at the rate
    `recovery` and speeds up one class. Pedestrians outside keep their speed.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the crowd, a distribution over rooms and speed classes.

        field:
            The path of the group of one member whose fixed density is the guidance of the exit signs.

        rooms:
            The number of rooms.

        speeds:
            The number of speed classes.

        recovery:
            The rate at which pedestrians speed up again.

    Examples:
        >>> from skcomplex.models.social.crowds import GuidanceInteraction
        >>> GuidanceInteraction('guidance', candidate='crowd', field='signs').recovery
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        rooms: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 7,
        speeds: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
        recovery: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.rooms = rooms
        self.speeds = speeds
        self.recovery = recovery

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the rate of the encounters of every class with the exit signs."""
        return crowding_table(int(self.rooms), int(self.speeds), 1)[0] * s.recovery

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table that moves a pedestrian up one speed class."""
        return crowding_table(int(self.rooms), int(self.speeds), 1)[1]
