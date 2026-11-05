"""Ageing of every triangle of a simplicial complex by one per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import HyperedgeInteraction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.specs import RoleSpec


class SimplexAgeingInteraction(HyperedgeInteraction):
    """Ageing of every triangle of a simplicial complex by one per step.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes.

        triangles:
            The path of the group on the triangles.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SimplexAgeingInteraction
        >>> SimplexAgeingInteraction('ageing', nodes='nodes', triangles='triangles').size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        triangles: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label, size=3)
        self.nodes = nodes
        self.triangles = triangles

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the ageing of the triangle."""
        return [UpdateLaw(s.hyperedge.age, s.hyperedge.age + 1)]
