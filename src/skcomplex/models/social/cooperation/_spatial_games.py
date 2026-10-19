"""Spatial games: cooperation on a lattice where players imitate more successful neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, ValidationError, parameter, state
from ....core.model import Element, LatticeSpace, Law, LocalRule, Neighborhood, Symbols, System
from ....core.runtime import register_model

type Array = npt.NDArray[np.float64]
type Offset = tuple[int, int]

IMITATIONS = ('best', 'fermi')
ORIGIN: Offset = (0, 0)
MOORE: tuple[Offset, ...] = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
VON_NEUMANN: tuple[Offset, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
SQUARE = Neighborhood(tuple((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)))
DIAMOND_RADIUS = 3
DIAMOND = Neighborhood(
    tuple(
        (dx, dy)
        for dx in range(-DIAMOND_RADIUS, DIAMOND_RADIUS + 1)
        for dy in range(-DIAMOND_RADIUS, DIAMOND_RADIUS + 1)
        if abs(dx) + abs(dy) <= DIAMOND_RADIUS
    ),
)


def _shift(first: Offset, second: Offset) -> Offset:
    """Return the sum of two offsets."""
    return first[0] + second[0], first[1] + second[1]


def _imitate_fermi(
    own: Array,
    own_payoff: Array,
    payoffs: list[Array],
    strategies: list[Array],
    *,
    noise: Any,  # noqa: ANN401
    random: Array,
) -> Array:
    """Return strategies copied from a random neighbour with probability `1 / (1 + exp((P_own - P_other) / noise))`."""
    count = len(payoffs)
    chosen = np.minimum((random * count).astype(int), count - 1)
    uniform = random * count - chosen
    picked = np.take_along_axis(np.stack(payoffs), chosen[None], axis=0)[0]
    strategy = np.take_along_axis(np.stack(strategies), chosen[None], axis=0)[0]
    exponent = np.clip((own_payoff - picked) / np.maximum(noise, 1e-12), -50, 50)
    return np.where(uniform < 1 / (1 + np.exp(exponent)), strategy, own)


def _imitate_best(own: Array, own_payoff: Array, payoffs: list[Array], strategies: list[Array]) -> Array:
    """Return the strategy of the player with the largest payoff among a player and its neighbours, itself on ties."""
    best, following = own_payoff.copy(), own.copy()
    for payoff, strategy in zip(payoffs, strategies, strict=True):
        following = np.where(payoff > best, strategy, following)
        best = np.maximum(best, payoff)
    return following


def _random_strategies(
    shape: Sequence[int],
    share: float,
    random_state: int | np.random.Generator | None,
) -> Array:
    """Return strategies that are 1, cooperation, with a probability and 0, defection, otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < share).astype(float)


class Prisoner(Element):
    """Player of the spatial prisoner's dilemma, a cooperator at 1 and a defector at 0.

    Every step each player plays the weak dilemma, with reward 1, temptation `temptation`, and 0 otherwise, against
    itself and its eight neighbours. It then copies the best player among them, or a random neighbour with the Fermi
    probability `1 / (1 + exp((P_own - P_other) / noise))`.

    Args:
        label:
            The label of the player or the lattice group.

        strategy:
            The initial strategy, 1 or 0.

        temptation:
            The payoff of a defector against a cooperator.

        noise:
            The noise of the Fermi imitation.

        imitation:
            `'best'` for the best neighbour, or `'fermi'` for the Fermi rule.

    Examples:
        >>> from skcomplex.models.social.cooperation import Prisoner
        >>> Prisoner('player').temptation
        1.85
    """

    strategy: float = state(default=1.0, domain=Domain.BOOLEAN)
    temptation: float = parameter(default=1.85, domain=Domain.NONNEGATIVE)
    noise: float = parameter(default=0.1, domain=Domain.POSITIVE)
    imitation: str = 'best'

    def laws(self, s: Symbols) -> list[Law]:
        """Return the play and imitation rule over the 5 by 5 square around a player.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ValidationError:
                The imitation is unknown.
        """
        imitation = self.imitation
        if imitation not in IMITATIONS:
            error_msg = f'Unknown imitation {imitation!r}. Use one of {", ".join(IMITATIONS)}.'
            raise ValidationError(error_msg)

        def rule(*values: Array, temptation: Any, noise: Any, random: Array) -> Array:  # noqa: ANN401
            cells = dict(zip(SQUARE.offsets, values, strict=True))

            def payoff(center: Offset) -> Array:
                own = cells[center]
                around = sum(cells[_shift(center, offset)] for offset in (*MOORE, ORIGIN))
                return own * around + (1 - own) * around * temptation

            payoffs = [payoff(offset) for offset in MOORE]
            strategies = [cells[offset] for offset in MOORE]
            if imitation == 'best':
                return _imitate_best(cells[ORIGIN], payoff(ORIGIN), payoffs, strategies)
            return _imitate_fermi(cells[ORIGIN], payoff(ORIGIN), payoffs, strategies, noise=noise, random=random)

        return [LocalRule(s.strategy, SQUARE, rule, stochastic=True)]


class Contributor(Element):
    """Player of the spatial public goods game, a cooperator at 1 and a defector at 0.

    Every player joins five groups, the one centred on itself and those centred on its four neighbours. In each group
    a cooperator pays 1, and the pot multiplied by `synergy` is shared by the five members. A player then copies a
    random neighbour with the Fermi probability `1 / (1 + exp((P_own - P_other) / noise))`.

    Args:
        label:
            The label of the player or the lattice group.

        strategy:
            The initial strategy, 1 or 0.

        synergy:
            The factor that multiplies the contributions of a group.

        noise:
            The noise of the Fermi imitation.

    Examples:
        >>> from skcomplex.models.social.cooperation import Contributor
        >>> Contributor('player').synergy
        4.5
    """

    strategy: float = state(default=1.0, domain=Domain.BOOLEAN)
    synergy: float = parameter(default=4.5, domain=Domain.NONNEGATIVE)
    noise: float = parameter(default=0.5, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the play and imitation rule over the sites within three steps of a player.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """

        def rule(*values: Array, synergy: Any, noise: Any, random: Array) -> Array:  # noqa: ANN401
            cells = dict(zip(DIAMOND.offsets, values, strict=True))
            group = (*VON_NEUMANN, ORIGIN)

            def payoff(center: Offset) -> Array:
                share = sum(
                    sum(cells[_shift(_shift(center, host), member)] for member in group) * synergy / len(group)
                    for host in group
                )
                return share - len(group) * cells[center]

            payoffs = [payoff(offset) for offset in VON_NEUMANN]
            strategies = [cells[offset] for offset in VON_NEUMANN]
            return _imitate_fermi(cells[ORIGIN], payoff(ORIGIN), payoffs, strategies, noise=noise, random=random)

        return [LocalRule(s.strategy, DIAMOND, rule, stochastic=True)]


@register_model(
    field='social/cooperation',
    tags=('lattice', 'games', 'cooperation', 'evolution'),
    references=('Nowak and May 1992', 'Szabó and Fáth 2007', 'Argun et al. 2021', 'Gros 2015'),
    defaults={'until': 50, 'dt': 1},
)
def spatial_prisoners_dilemma(
    shape: Sequence[int] = (50, 50),
    *,
    temptation: float = 1.85,
    imitation: str = 'best',
    noise: float = 0.1,
    share: float = 0.9,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the spatial prisoner's dilemma of Nowak and May, with best-neighbour or Fermi imitation.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        temptation:
            The payoff of a defector against a cooperator.

        imitation:
            `'best'` to copy the best neighbour, or `'fermi'` to copy a random neighbour with the Fermi probability.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The model, with the strategies `players.strategy`.

    Examples:
        >>> from skcomplex.models.social.cooperation import spatial_prisoners_dilemma
        >>> spatial_prisoners_dilemma((10, 10)).elements[0].size
        100
    """
    players = Prisoner.lattice(
        'players',
        LatticeSpace(tuple(shape)),
        strategy=_random_strategies(shape, share, random_state),
        temptation=temptation,
        noise=noise,
        imitation=imitation,
    )
    return System(elements=[players])


@register_model(
    field='social/cooperation',
    tags=('lattice', 'games', 'cooperation', 'public goods'),
    references=('Szabó and Hauert 2002', 'Szabó and Fáth 2007'),
    defaults={'until': 50, 'dt': 1},
)
def public_goods_game(
    shape: Sequence[int] = (50, 50),
    *,
    synergy: float = 4.5,
    noise: float = 0.5,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the spatial public goods game in groups of five neighbours, with Fermi imitation.

    Args:
        shape:
            The number of rows and columns of the periodic lattice.

        synergy:
            The factor that multiplies the contributions of a group.

        noise:
            The noise of the Fermi imitation.

        share:
            The share of cooperators at the start.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The model, with the strategies `players.strategy`.

    Examples:
        >>> from skcomplex.models.social.cooperation import public_goods_game
        >>> public_goods_game((10, 10)).elements[0].size
        100
    """
    strategies = _random_strategies(shape, share, random_state)
    players = Contributor.lattice(
        'players',
        LatticeSpace(tuple(shape)),
        strategy=strategies,
        synergy=synergy,
        noise=noise,
    )
    return System(elements=[players])
