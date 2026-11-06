"""Activity of every node in its phase, 1 in the giant component of the active links and 0 outside it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import components
from ....core.specs import RoleSpec


class GiantComponentInteraction(Interaction):
    """Activity of every node in its phase, 1 in the giant component of the active links and 0 outside it.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes, whose activity the interaction sets.

        links:
            The path of the group on the edges, whose activity selects the links of the components.

        phase:
            The phase of the step the interaction runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import GiantComponentInteraction
        >>> GiantComponentInteraction('percolation', nodes='nodes', links='links').links
        'links'
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.phase = phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the membership of the giant component of the active links."""
        return [UpdateLaw(s.nodes.active, components(s.links.active, reduce='giant'), phase=self.phase)]
