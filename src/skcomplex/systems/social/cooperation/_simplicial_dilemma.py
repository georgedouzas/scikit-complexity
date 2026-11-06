"""Play of two-player games along the links of the triangles of a simplicial complex, which sets the payoffs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import count, total, where
from ....core.specs import ParameterSpec, RoleSpec

PAIR = 2


class SimplicialDilemmaInteraction(CouplingInteraction):
    """Play of two-player games along the links of the triangles of a simplicial complex, which sets the payoffs.

    It is a `CouplingInteraction` over the hyperedges of a network in the map form, a cooperator at 1 and a defector
    at 0. Every game pays 1 for mutual cooperation, 0 for mutual defection, and the sucker's payoff `S` and the
    temptation `T` when the strategies differ. A triangle of three links is three hyperedges of two members, along
    which the players play game 1 of `S1` and `T1`. A triangle filled as a 2-simplex is one hyperedge of three members,
    where a player plays each other member while the third looks on: game 2 of `S2` and `T2` when the third member
    shares the player's strategy, and game 3 of `S3` and `T3` otherwise. The payoff of a player is the sum over the two
    links of every triangle around it divided by twice its number of triangles, a parameter of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the network.

        sucker:
            The payoff `S1` of a cooperator against a defector along a link.

        temptation:
            The payoff `T1` of a defector against a cooperator along a link.

        sucker_allied:
            The payoff `S2` in a 2-simplex whose third member shares the player's strategy.

        temptation_allied:
            The payoff `T2` in a 2-simplex whose third member shares the player's strategy.

        sucker_opposed:
            The payoff `S3` in a 2-simplex whose third member holds the other strategy.

        temptation_opposed:
            The payoff `T3` in a 2-simplex whose third member holds the other strategy.

        phase:
            The phase of the step the play runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.cooperation import SimplicialDilemmaInteraction
        >>> SimplicialDilemmaInteraction('play', target='players', temptation_allied=0.8).temptation_allied
        0.8
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        sucker: Annotated[float, ParameterSpec()] = -0.2,
        temptation: Annotated[float, ParameterSpec()] = 1.2,
        sucker_allied: Annotated[float, ParameterSpec()] = 0.5,
        temptation_allied: Annotated[float, ParameterSpec()] = 0.8,
        sucker_opposed: Annotated[float, ParameterSpec()] = -0.2,
        temptation_opposed: Annotated[float, ParameterSpec()] = 1.2,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            mode='hyperedges',
            form='map',
            write='payoff',
            function=self._get_games,
            combine=self._get_mean,
            phase=phase,
        )
        self.sucker = sucker
        self.temptation = temptation
        self.sucker_allied = sucker_allied
        self.temptation_allied = temptation_allied
        self.sucker_opposed = sucker_opposed
        self.temptation_opposed = temptation_opposed

    def _get_games(self, hyperedge: SimpleNamespace) -> Expression:
        """Return the payoff of a member from the games along its links in one hyperedge of two or three members.

        In a link a cooperator earns 1 or `S1` and a defector `T1` or 0 against a cooperator or a defector. In a
        2-simplex with `c` cooperating others a cooperator earns 2, `1 + S2`, or `2 S3`, and a defector `2 T3`, `T2`, or
        0, for `c` of 2, 1, or 0.
        """
        own = hyperedge.member.strategy
        others = total(hyperedge.members.strategy, over=hyperedge.members) - own
        link = own * (others + (1 - others) * hyperedge.sucker) + (1 - own) * others * hyperedge.temptation
        cooperator = where(
            others == PAIR,
            PAIR,
            where(others == 1, 1 + hyperedge.sucker_allied, PAIR * hyperedge.sucker_opposed),
        )
        defector = where(
            others == PAIR,
            PAIR * hyperedge.temptation_opposed,
            where(others == 1, hyperedge.temptation_allied, 0),
        )
        simplex = own * cooperator + (1 - own) * defector
        return where(count(over=hyperedge.members) == PAIR, link, simplex)

    @staticmethod
    def _get_mean(player: SimpleNamespace, games: Expression) -> Expression:
        """Return the payoffs of the games of a player divided by twice its number of triangles."""
        return games / (PAIR * player.member.triangles)
