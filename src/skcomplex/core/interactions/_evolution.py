"""Define evolution interactions, which breed the strategies of a group by selection, crossover, and mutation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
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

SELECTIONS = ('tournament', 'roulette', 'rank')
REPLACEMENTS = ('generational', 'moran', 'death-birth')
MIN_ALPHABET = 2


class EvolutionInteraction(Interaction):
    """Replacement of the strategies of an agent group by offspring selected on a fitness, with crossover and mutation.

    Every step, or every round of its phase, the interaction selects parents by tournaments of `tournament` entrants,
    by roulette with weights of the fitness minus its minimum, or by rank with weights `1` for the least fit to `n` for
    the fittest of `n` members. The `'generational'` replacement selects as many parents as there are members, each
    pair of which exchanges the tails after a random cut with probability `crossover`, and passes the `elite` fittest
    strategies unchanged. The `'moran'` replacement runs as many birth-death events as there are members, one after the
    other, each copying a selected parent onto a member drawn uniformly, and the copy takes the fitness of its parent
    for the later events of the step. The `'death-birth'` replacement removes every member with probability `death`
    and fills its place with the offspring of a parent selected among the survivors, or among all members when none
    survives. Offspring mutate: each gene with probability `mutation` to another symbol of an alphabet of `alphabet`
    symbols, or by Gaussian noise of deviation `scale` when `alphabet` is `None`, or by `mutation(strategy, rng)` when
    `mutation` is a function. The draws come from the random stream of the model.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the agent group whose strategies evolve.

        strategy:
            The name of the state of the members that holds the strategy, a sequence of genes.

        fitness:
            The name of the state of the members that holds the fitness.

        selection:
            The selection of parents, `'tournament'`, `'roulette'`, or `'rank'`.

        replacement:
            The replacement of strategies, `'generational'`, `'moran'`, or `'death-birth'`.

        crossover:
            The probability that a pair of parents exchanges the tails of their strategies, in the generational
            replacement.

        mutation:
            The probability that a gene mutates, or a function `mutation(strategy, rng)` that returns the mutated copy
            of a strategy.

        elite:
            The number of fittest strategies copied unchanged, in the generational replacement.

        tournament:
            The number of entrants of every tournament.

        alphabet:
            The number of symbols of a discrete strategy, or `None` for a real strategy.

        scale:
            The standard deviation of the mutation of a real strategy.

        death:
            The probability that a member is replaced in a step, in the death-birth replacement.

        phase:
            The phase of the step the generation runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The selection or the replacement is unknown, a probability is outside `[0, 1]`, the mutation is neither a
            probability nor a function, or a count is invalid.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.core.interactions import EvolutionInteraction
        >>> evolution = EvolutionInteraction('evolution', target='agents', selection='roulette', phase='breeding')
        >>> evolution.check().prepare().run(SimpleNamespace()).laws_[0].phase
        'breeding'
        >>> EvolutionInteraction('evolution', target='agents', selection='rank', replacement='moran').check().selection
        'rank'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        strategy: str = 'strategy',
        fitness: str = 'fitness',
        selection: str = 'tournament',
        replacement: str = 'generational',
        crossover: float = 0.7,
        mutation: float | Callable[[Array, np.random.Generator], Array] = 0.01,
        elite: int = 0,
        tournament: int = 2,
        alphabet: int | None = 2,
        scale: float = 0.1,
        death: float = 0.1,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.strategy = strategy
        self.fitness = fitness
        self.selection = selection
        self.replacement = replacement
        self.crossover = crossover
        self.mutation = mutation
        self.elite = elite
        self.tournament = tournament
        self.alphabet = alphabet
        self.scale = scale
        self.death = death
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

    def _check_probabilities(self) -> None:
        """Raise an error for a probability outside `[0, 1]` or a mutation neither a probability nor callable."""
        probabilities = [('crossover', self.crossover), ('death', self.death)]
        if not callable(self.mutation):
            probabilities.append(('mutation', self.mutation))
        for name, value in probabilities:
            if not isinstance(value, int | float) or not 0 <= value <= 1:
                error_msg = f'The {name} probability of {self.label!r} is {value!r}. Give a value in [0, 1].'
                raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for an unknown variant, a probability outside `[0, 1]`, or an invalid count.

        Raises:
            ValidationError:
                The type declares states, the selection or the replacement is unknown, a probability is outside
                `[0, 1]`, the mutation is neither a probability nor a function, or a count is invalid.
        """
        super()._check()
        for name, value, variants in (
            ('selection', self.selection, SELECTIONS),
            ('replacement', self.replacement, REPLACEMENTS),
        ):
            if value not in variants:
                error_msg = f'Unknown {name} {value!r} of {self.label!r}. Use one of {variants}.'
                raise ValidationError(error_msg)
        self._check_probabilities()
        self._check_counts()

    def _select(self, fitness: Array, count: int, rng: np.random.Generator) -> npt.NDArray[np.intp]:
        """Return the indices of a number of parents selected on their fitness."""
        size = len(fitness)
        if self.selection == 'tournament':
            entrants = rng.integers(size, size=(count, self.tournament))
            return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]
        if self.selection == 'rank':
            weights = np.empty(size)
            weights[np.argsort(fitness, kind='stable')] = np.arange(1, size + 1)
        else:
            weights = fitness - fitness.min()
            weights = weights if weights.sum() else np.ones_like(weights)
        return rng.choice(size, size=count, p=weights / weights.sum())

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
        """Return the children with every gene mutated with the mutation probability, or by the mutation function."""
        if callable(self.mutation):
            mutation = self.mutation
            return np.array([np.ravel(mutation(child.copy(), rng)) for child in children], dtype=float)
        mutated = rng.random(children.shape) < self.mutation
        if self.alphabet is None:
            return children + mutated * rng.normal(0.0, self.scale, children.shape)
        shift = rng.integers(1, self.alphabet, children.shape)
        return np.where(mutated, (children + shift) % self.alphabet, children)

    def _breed_generation(self, flat: Array, fitness: Array, rng: np.random.Generator) -> Array:
        """Return the strategies of the next generation, one row per member."""
        children = self._mutate(self._cross(flat[self._select(fitness, len(flat), rng)], rng), rng)
        if self.elite:
            best = np.argsort(-fitness, kind='stable')[: self.elite]
            children[: len(best)] = flat[best]
        return children

    def _breed_moran(self, flat: Array, fitness: Array, rng: np.random.Generator) -> Array:
        """Return the strategies after one birth-death event per member, one after the other."""
        strategies, fitness = flat.copy(), fitness.copy()
        for _ in range(len(strategies)):
            parent = int(self._select(fitness, 1, rng)[0])
            dying = int(rng.integers(len(strategies)))
            strategies[dying] = self._mutate(strategies[parent : parent + 1], rng)[0]
            fitness[dying] = fitness[parent]
        return strategies

    def _breed_death_birth(self, flat: Array, fitness: Array, rng: np.random.Generator) -> Array:
        """Return the strategies after the dead members are filled by the offspring of the survivors."""
        dead = rng.random(len(flat)) < self.death
        if not dead.any():
            return flat
        survivors = np.flatnonzero(~dead) if not dead.all() else np.arange(len(flat))
        parents = survivors[self._select(fitness[survivors], int(dead.sum()), rng)]
        strategies = flat.copy()
        strategies[dead] = self._mutate(flat[parents], rng)
        return strategies

    def _breed(self, strategies: Array, fitness: Array, rng: np.random.Generator) -> Array:
        """Return the strategies after the replacement, one row per member."""
        flat = strategies.reshape(len(strategies), -1)
        if self.replacement == 'moran':
            children = self._breed_moran(flat, fitness, rng)
        elif self.replacement == 'death-birth':
            children = self._breed_death_birth(flat, fitness, rng)
        else:
            children = self._breed_generation(flat, fitness, rng)
        return children.reshape(strategies.shape)

    def _evolve(self, model: ModelViewProtocol) -> None:
        """Replace the strategies of the members by their offspring."""
        agents = model.agents('target')
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
