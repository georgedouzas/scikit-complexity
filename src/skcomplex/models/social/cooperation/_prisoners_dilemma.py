"""Play of the weak prisoner's dilemma between neighbours, and imitation of more successful neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.interactions import LocalRuleInteraction
from ....core.spaces import LatticeNeighborhood
from ....errors import ValidationError
from ....specs import RoleSpec
from ....types import Array, Pair

IMITATIONS = ('best', 'fermi')


ORIGIN: Pair = (0, 0)


MOORE: tuple[Pair, ...] = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


SQUARE = LatticeNeighborhood(tuple((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)))


class PrisonersDilemmaInteraction(LocalRuleInteraction):
    """Play of the weak prisoner's dilemma between neighbours, and imitation of more successful neighbours.

    It is a stochastic `LocalRuleInteraction` of the strategies over the 5 by 5 square around a player, a cooperator
    at 1 and a defector at 0. Every step each player plays the weak dilemma, with reward 1, temptation `temptation`,
    and 0 otherwise, against itself and its eight neighbours. It then copies the best player among them, or a random
    neighbour with the Fermi probability `1 / (1 + exp((P_own - P_other) / noise))`. The temptation and the noise are
    parameters of the players.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the players on the lattice.

        imitation:
            `'best'` for the best neighbour, or `'fermi'` for the Fermi rule.

    Raises:
        ValidationError:
            The imitation is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.models.social.cooperation import PrisonersDilemmaInteraction
        >>> PrisonersDilemmaInteraction('play', target='players', imitation='fermi').imitation
        'fermi'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], imitation: str = 'best') -> None:
        super().__init__(label, target=target, state='strategy', rule=self._play, neighborhood=SQUARE, stochastic=True)
        self.imitation = imitation

    def _check(self) -> None:
        """Raise an error for an unknown imitation.

        Raises:
            ValidationError:
                The imitation is unknown.
        """
        super()._check()
        if self.imitation not in IMITATIONS:
            error_msg = f'Unknown imitation {self.imitation!r}. Use one of {", ".join(IMITATIONS)}.'
            raise ValidationError(error_msg)

    def _play(self, *values: Array, temptation: float | Array, noise: float | Array, random: Array) -> Array:
        """Return the strategies after the play of every player against its neighbours and the imitation."""
        cells = dict(zip(SQUARE.offsets, values, strict=True))

        def payoff(center: Pair) -> Array:
            own = cells[center]
            around = sum(cells[PrisonersDilemmaInteraction._shift(center, offset)] for offset in (*MOORE, ORIGIN))
            return own * around + (1 - own) * around * temptation

        payoffs = [payoff(offset) for offset in MOORE]
        strategies = [cells[offset] for offset in MOORE]
        if self.imitation == 'best':
            return PrisonersDilemmaInteraction._imitate_best(cells[ORIGIN], payoff(ORIGIN), payoffs, strategies)
        return PrisonersDilemmaInteraction._imitate_fermi(
            cells[ORIGIN],
            payoff(ORIGIN),
            payoffs,
            strategies,
            noise=noise,
            random=random,
        )

    @staticmethod
    def _shift(first: Pair, second: Pair) -> Pair:
        """Return the sum of two offsets."""
        return first[0] + second[0], first[1] + second[1]

    @staticmethod
    def _imitate_fermi(
        own: Array,
        own_payoff: Array,
        payoffs: list[Array],
        strategies: list[Array],
        *,
        noise: float | Array,
        random: Array,
    ) -> Array:
        """Return the strategy of a random neighbour with probability `1 / (1 + exp((P_own - P_other) / noise))`."""
        count = len(payoffs)
        chosen = np.minimum((random * count).astype(int), count - 1)
        uniform = random * count - chosen
        picked = np.take_along_axis(np.stack(payoffs), chosen[None], axis=0)[0]
        strategy = np.take_along_axis(np.stack(strategies), chosen[None], axis=0)[0]
        exponent = np.clip((own_payoff - picked) / np.maximum(noise, 1e-12), -50, 50)
        return np.where(uniform < 1 / (1 + np.exp(exponent)), strategy, own)

    @staticmethod
    def _imitate_best(own: Array, own_payoff: Array, payoffs: list[Array], strategies: list[Array]) -> Array:
        """Return the strategy of the largest payoff among a player and its neighbours, its own on ties."""
        best, following = own_payoff.copy(), own.copy()
        for payoff, strategy in zip(payoffs, strategies, strict=True):
            following = np.where(payoff > best, strategy, following)
            best = np.maximum(best, payoff)
        return following
