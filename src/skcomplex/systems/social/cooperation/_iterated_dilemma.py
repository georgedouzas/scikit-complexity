"""Play of the iterated prisoner's dilemma between neighbours, revision to the best strategy, and mutation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import LatticeNeighborhood
from ....core.specs import RoleSpec
from ....core.types import Array, Pair

ORIGIN: Pair = (0, 0)


VON_NEUMANN: tuple[Pair, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


DIAMOND = LatticeNeighborhood(
    tuple((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) <= 2),
)


class IteratedDilemmaInteraction(LocalRuleInteraction):
    """Play of the iterated prisoner's dilemma between neighbours, revision to the best strategy, and mutation.

    It is a stochastic `LocalRuleInteraction` of the strategies over the cells within two von Neumann steps of a
    player. Every step each player plays the `N`-round prisoner's dilemma with its four von Neumann neighbours and sums
    its years in prison. A player of strategy `n` against one of strategy `m < n` gets `m R + S + (N - m - 1) P`, the
    other `m R + T + (N - m - 1) P`, and two players of strategy `n` get `n R + (N - n) P`. Every player then adopts
    the strategy of the lowest score among itself and its four neighbours, a random one of them on ties, and finally
    mutates to a uniformly drawn strategy with the probability `mu`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

    Examples:
        >>> from skcomplex.systems.social.cooperation import IteratedDilemmaInteraction
        >>> IteratedDilemmaInteraction('play', target='players').state
        'strategy'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, state='strategy', rule=self._play, neighborhood=DIAMOND, stochastic=True)

    @staticmethod
    def years(
        own: Array,
        other: Array,
        *,
        rounds: float | Array,
        reward: float | Array,
        sucker: float | Array,
        temptation: float | Array,
        punishment: float | Array,
    ) -> Array:
        """Return the years in prison of a player of strategy `n` against one of strategy `m` in an `N`-round game.

        Args:
            own:
                The strategies `n` of the players.

            other:
                The strategies `m` of their opponents.

            rounds:
                The number `N` of rounds.

            reward:
                The punishment `R` of mutual cooperation.

            sucker:
                The punishment `S` of a betrayed cooperator.

            temptation:
                The punishment `T` of a defector who betrays.

            punishment:
                The punishment `P` of mutual defection.

        Returns:
            years:
                The years in prison of every player.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.systems.social.cooperation import IteratedDilemmaInteraction
            >>> years = IteratedDilemmaInteraction.years(
            ...     np.arange(11.0), np.full(11, 6.0), rounds=10, reward=0.5, sucker=1.5, temptation=0.0, punishment=1.0
            ... )
            >>> int(np.argmin(years))
            5
        """
        first = np.minimum(own, other)
        betrayal = np.where(own < other, temptation, sucker)
        mixed = first * reward + betrayal + (rounds - first - 1) * punishment
        return np.where(own == other, own * reward + (rounds - own) * punishment, mixed)

    def _play(
        self,
        *values: Array,
        rounds: float | Array,
        reward: float | Array,
        sucker: float | Array,
        temptation: float | Array,
        punishment: float | Array,
        mutation: float | Array,
        extremes: float | Array,
        random: Array,
    ) -> Array:
        """Return the strategies after the play against the neighbours, the revision, and the mutation."""
        cells = dict(zip(DIAMOND.offsets, values, strict=True))
        payoffs = {'rounds': rounds, 'reward': reward, 'sucker': sucker, 'temptation': temptation}

        def score(center: Pair) -> Array:
            own = cells[center]
            return sum(
                (
                    IteratedDilemmaInteraction.years(
                        own,
                        cells[center[0] + dx, center[1] + dy],
                        punishment=punishment,
                        **payoffs,
                    )
                    for dx, dy in VON_NEUMANN
                ),
                start=np.zeros_like(own),
            )

        candidates = (ORIGIN, *VON_NEUMANN)
        scores = np.stack([score(offset) for offset in candidates])
        strategies = np.stack([cells[offset] for offset in candidates])
        tied = np.isclose(scores, scores.min(axis=0))
        mutation = np.broadcast_to(mutation, random.shape)
        choice = np.clip((random - mutation) / np.maximum(1 - mutation, 1e-12), 0.0, 1.0 - 1e-12)
        rank = (choice * tied.sum(axis=0)).astype(int)
        picked = np.argmax(np.cumsum(tied, axis=0) > rank, axis=0)
        revised = np.take_along_axis(strategies, picked[None], axis=0)[0]
        draw = random / np.maximum(mutation, 1e-12)
        mutant = np.where(
            np.broadcast_to(extremes, random.shape) > 0,
            np.where(draw < 0.5, 0.0, rounds),
            np.minimum(np.floor(draw * (np.asarray(rounds) + 1)), rounds),
        )
        return np.where(random < mutation, mutant, revised)
