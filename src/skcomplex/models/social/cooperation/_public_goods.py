"""Play of the public goods game in the groups of a player and its four neighbours, which sets the payoffs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import GroupingInteraction
from ....core.laws.utils import count, total
from ....specs import RoleSpec


class PublicGoodsInteraction(GroupingInteraction):
    """Play of the public goods game in the groups of a player and its four neighbours, which sets the payoffs.

    It is a symbolic `GroupingInteraction` of the von Neumann neighbourhoods of a lattice, a cooperator at 1 and a
    defector at 0. Every player joins five groups, the one centred on itself and those centred on its four neighbours.
    In each group a cooperator pays 1, and the pot multiplied by `synergy` is shared by the members, so the payoff of a
    player is the sum of its shares minus its contributions. The synergy is a parameter of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

        phase:
            The phase of the step the play runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.cooperation import PublicGoodsInteraction
        >>> PublicGoodsInteraction('play', target='players').write
        'payoff'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            groups='neighborhood',
            form='symbolic',
            write='payoff',
            function=self._get_share,
            phase=phase,
        )

    def _get_share(self, group: SimpleNamespace) -> Expression:
        """Return the share of the pot of a group for its member, minus the contribution of the member."""
        pot = group.member.synergy * total(group.members.strategy, over=group.members)
        return pot / count(over=group.members) - group.member.strategy
