"""Moves of a random walker across a hyperedge, to every other member with weight `|e| - 1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import HyperedgeInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total
from ....core.specs import RoleSpec


class SizeBiasedMoveInteraction(HyperedgeInteraction):
    """Moves of a random walker across a hyperedge, to every other member with weight `|e| - 1`.

    A walker at node `i` moves to node `j` at rate `T_ij = k_ij / D_i`, where `k_ij = sum_e (|e| - 1)` over the
    hyperedges `e` that hold both, and `D_i = sum_j k_ij`. On a hyperedge, the focal member gains `(|e| - 1) p_j / D_j`
    from every other member `j` and loses `(|e| - 1)^2 p_i / D_i`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of walker densities on the nodes.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SizeBiasedMoveInteraction
        >>> SizeBiasedMoveInteraction('moves', target='nodes').target
        'nodes'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flow into and out of the focal member across one hyperedge."""
        weight = len(s.others)
        member = s.member
        inflow = total(s.members.p / s.members.D, over=s.members) - member.p / member.D
        return [RateLaw(member.p, weight * inflow - weight**2 * member.p / member.D)]
