"""Weekly meal of every cow, which eats its bite of the grass of the patch in its cell."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, cast

from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.specs import RoleSpec
from ._cow_view import CowView
from ._pasture_view import PastureView


class GrazingInteraction(TransferInteraction):
    """Weekly meal of every cow, which eats its bite of the grass of the patch in its cell.

    It is a transfer of the grass of every patch, the bite of the cow in its cell or what grass is left, to the
    environment.

    Args:
        label:
            The label of the interaction.

        cows:
            The path of the group of cows.

        pasture:
            The path of the group of patches.

    Examples:
        >>> from skcomplex.systems.biology.ecology import GrazingInteraction
        >>> GrazingInteraction('grazing', cows='cows', pasture='pasture').pasture
        'pasture'
    """

    def __init__(
        self,
        label: str,
        *,
        cows: Annotated[str, RoleSpec()],
        pasture: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            source=pasture,
            target=cows,
            state='grass',
            into=Sink(),
            amount=self._get_bite,
            pairing='cell',
            form='agent',
        )
        self.cows = cows
        self.pasture = pasture

    def _get_role(self, role: str) -> str:
        """Return the patches as the source of the grass and the cows as its target."""
        return {'source': 'pasture', 'target': 'cows'}.get(role, role)

    @staticmethod
    def _get_bite(patch: AgentViewProtocol, cow: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the grass a cow eats from the patch of its cell, its bite or what grass is left."""
        del system
        return min(cast(PastureView, patch).grass, cast(CowView, cow).bite)
