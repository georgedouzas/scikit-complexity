"""Infection of every resident whose resistance is below the strength of the season's strain, who becomes immune."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.specs import RoleSpec


class ResistanceInfectionInteraction(Interaction):
    """Infection of every resident whose resistance is below the strength of the season's strain, who becomes immune.

    A resident with `R < eta` is infected in the season and its resistance becomes 1, so it is never infected again.

    Args:
        label:
            The label of the interaction.

        residents:
            The path of the group of residents.

        pathogen:
            The path of the pathogen.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ResistanceInfectionInteraction
        >>> ResistanceInfectionInteraction('infection', residents='people', pathogen='flu').pathogen
        'flu'
    """

    def __init__(
        self,
        label: str,
        *,
        residents: Annotated[str, RoleSpec()],
        pathogen: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.residents = residents
        self.pathogen = pathogen

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection and the immunity of the residents below the strength of the strain.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        resident, strength = s.residents, s.pathogen.eta
        susceptible = strength > resident.R
        return [
            UpdateLaw(resident.R, where(susceptible, 1, resident.R)),
            UpdateLaw(resident.infected, where(susceptible, 1, 0)),
        ]
