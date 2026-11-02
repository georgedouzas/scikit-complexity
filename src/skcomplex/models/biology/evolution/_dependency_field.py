"""Field of every species, the sum of the couplings `w` of its edges to the species it depends on."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import neighbors
from ....specs import RoleSpec


class DependencyFieldInteraction(Interaction):
    """Field of every species, the sum of the couplings `w` of its edges to the species it depends on.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species on the nodes.

        couplings:
            The path of the variable group on the edges.

    Examples:
        >>> from skcomplex.models.biology.evolution import DependencyFieldInteraction
        >>> DependencyFieldInteraction('field', species='species', couplings='couplings').couplings
        'couplings'
    """

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[str, RoleSpec()],
        couplings: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.species = species
        self.couplings = couplings

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the field of every species.

        Args:
            s:
                The namespace with `s.species` and `s.couplings`.

        Returns:
            laws:
                The update of the field.
        """
        return [UpdateLaw(s.species.h, neighbors(1, weight=s.couplings.w))]
