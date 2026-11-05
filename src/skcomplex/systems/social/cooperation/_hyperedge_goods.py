"""Play of the public goods game in every hyperedge around a player, which sets the payoffs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import count, total
from ....core.specs import RoleSpec


class HyperedgeGoodsInteraction(CouplingInteraction):
    """Play of the public goods game in every hyperedge around a player, which sets the payoffs.

    It is a `CouplingInteraction` over the hyperedges of a network in the map form, a cooperator at 1 and a defector
    at 0. In every hyperedge of `m` members a cooperator pays 1, and the pot multiplied by `synergy` is shared by the
    `m` members, so the payoff of a player is the sum over its hyperedges of its share minus its contribution. The
    synergy is a parameter of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the network.

        phase:
            The phase of the step the play runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.cooperation import HyperedgeGoodsInteraction
        >>> HyperedgeGoodsInteraction('play', target='players').write
        'payoff'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], phase: str | None = None) -> None:
        super().__init__(
            label,
            target=target,
            mode='hyperedges',
            form='map',
            write='payoff',
            function=self._get_share,
            phase=phase,
        )

    def _get_share(self, hyperedge: SimpleNamespace) -> Expression:
        """Return the share of the pot of a hyperedge for its member, minus the contribution of the member."""
        pot = hyperedge.member.synergy * total(hyperedge.members.strategy, over=hyperedge.members)
        return pot / count(over=hyperedge.members) - hyperedge.member.strategy
