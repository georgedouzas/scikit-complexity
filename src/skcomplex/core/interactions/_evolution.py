"""Define evolution interactions, which breed the strategies of a group by selection, crossover, and mutation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import Array
from ..laws import BaseLaw, RuleLaw
from ..types import ModelViewProtocol
from ._interaction import Interaction

SELECTIONS = ('tournament', 'roulette')
MIN_ALPHABET = 2


class EvolutionInteraction(Interaction):
    """Generation of strategies of an agent group by selection on a fitness, one-point crossover, and mutation.

    Every step, or every round of its phase, the interaction selects as many parents as there are members, by
    tournaments of `tournament` entrants or by roulette with weights of the fitness minus its minimum. Each pair of
    parents exchanges the tails after a random cut with probability `crossover`, and each gene mutates with
    probability `mutation`, to another symbol of an alphabet of `alphabet` symbols, or by Gaussian noise of deviation
    `scale` when `alphabet` is `None`. The `elite` fittest strategies pass unchanged. The draws come from the random
    stream of the model.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        members:
            The path of the agent group whose strategies evolve.

        strategy:
            The name of the state of the members that holds the strategy, a sequence of genes.

        fitness:
            The name of the state of the members that holds the fitness.

        selection:
            The selection of parents, `'tournament'` or `'roulette'`.

        crossover:
            The probability that a pair of parents exchanges the tails of their strategies.

        mutation:
            The probability that a gene mutates.

        elite:
            The number of fittest strategies copied unchanged.

        tournament:
            The number of entrants of every tournament.

        alphabet:
            The number of symbols of a discrete strategy, or `None` for a real strategy.

        scale:
            The standard deviation of the mutation of a real strategy.

        phase:
            The phase of the step the generation runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The selection is unknown, a probability is outside `[0, 1]`, or a count is invalid.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.core.interactions import EvolutionInteraction
        >>> evolution = EvolutionInteraction('evolution', members='agents', selection='roulette', phase='breeding')
        >>> evolution.check().prepare().run(SimpleNamespace()).laws_[0].phase
        'breeding'
    """

    def __init__(
        self,
        label: str,
        *,
        members: Annotated[str, RoleSpec()],
        strategy: str = 'strategy',
        fitness: str = 'fitness',
        selection: str = 'tournament',
        crossover: float = 0.7,
        mutation: float = 0.01,
        elite: int = 0,
        tournament: int = 2,
        alphabet: int | None = 2,
        scale: float = 0.1,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.members = members
        self.strategy = strategy
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.elite = elite
        self.tournament = tournament
        self.alphabet = alphabet
        self.scale = scale
        self.phase = phase

    def _check_counts(self) -> None:
        """Raise an error for a negative elite, a tournament without entrants, or an alphabet of one symbol."""
        if self.elite < 0 or self.tournament < 1:
            error_msg = (
                f'{self.label!r} has elite {self.elite} and tournament {self.tournament}. Give an elite of at least 0 '
                'and a tournament of at least 1.'
            )
            raise ValidationError(error_msg)
        if self.alphabet is not None and self.alphabet < MIN_ALPHABET:
            error_msg = (
                f'{self.label!r} has an alphabet of {self.alphabet} symbols. Give at least {MIN_ALPHABET}, or `None`.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown selection, a probability outside `[0, 1]`, or an invalid count.

        Raises:
            ValidationError:
                The type declares states, the selection is unknown, a probability is outside `[0, 1]`, or a count is
                invalid.
        """
        super()._check()
        if self.selection not in SELECTIONS:
            error_msg = f'Unknown selection {self.selection!r} of {self.label!r}. Use one of {SELECTIONS}.'
            raise ValidationError(error_msg)
        for name, value in (('crossover', self.crossover), ('mutation', self.mutation)):
            if not 0 <= value <= 1:
                error_msg = f'The {name} probability of {self.label!r} is {value}. Give a value in [0, 1].'
                raise ValidationError(error_msg)
        self._check_counts()

    def _select(self, fitness: Array, rng: np.random.Generator) -> npt.NDArray[np.intp]:
        """Return the indices of the parents, one per member."""
        count = len(fitness)
        if self.selection == 'tournament':
            entrants = rng.integers(count, size=(count, self.tournament))
            return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]
        weights = fitness - fitness.min()
        weights = weights if weights.sum() else np.ones_like(weights)
        return rng.choice(count, size=count, p=weights / weights.sum())

    def _cross(self, parents: Array, rng: np.random.Generator) -> Array:
        """Return the parents after one-point crossover of every consecutive pair."""
        children = parents.copy()
        length = children.shape[1]
        for first in range(0, len(children) - 1, 2):
            if length > 1 and rng.random() < self.crossover:
                cut = int(rng.integers(1, length))
                tail = children[first, cut:].copy()
                children[first, cut:] = children[first + 1, cut:]
                children[first + 1, cut:] = tail
        return children

    def _mutate(self, children: Array, rng: np.random.Generator) -> Array:
        """Return the children with every gene mutated with the mutation probability."""
        mutated = rng.random(children.shape) < self.mutation
        if self.alphabet is None:
            return children + mutated * rng.normal(0.0, self.scale, children.shape)
        shift = rng.integers(1, self.alphabet, children.shape)
        return np.where(mutated, (children + shift) % self.alphabet, children)

    def _breed(self, strategies: Array, fitness: Array, rng: np.random.Generator) -> Array:
        """Return the strategies of the next generation, one row per member."""
        flat = strategies.reshape(len(strategies), -1)
        children = self._mutate(self._cross(flat[self._select(fitness, rng)], rng), rng)
        if self.elite:
            best = np.argsort(-fitness, kind='stable')[: self.elite]
            children[: len(best)] = flat[best]
        return children.reshape(strategies.shape)

    def _evolve(self, model: ModelViewProtocol) -> None:
        """Replace the strategies of the members by the next generation."""
        agents = model.agents('members')
        if not agents:
            return
        strategies = np.array([getattr(agent, self.strategy) for agent in agents], dtype=float)
        fitness = np.array([getattr(agent, self.fitness) for agent in agents], dtype=float)
        for agent, child in zip(agents, self._breed(strategies, fitness, model.rng), strict=True):
            setattr(agent, self.strategy, child)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of one generation.

        Args:
            namespace:
                The namespace of the symbols the laws can use, which the rule does not read.

        Returns:
            laws:
                The rule that breeds the next generation.
        """
        return [RuleLaw(self._evolve, phase=self.phase)]
