"""Define the interaction that breeds strategies by selection and mutation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.errors import SimulationError, ValidationError
from ....core.specs import BaseSpec, ParameterSpec, RoleSpec
from ....core.types import Array, ElementKind

SELECTIONS = ('tournament', 'roulette', 'proportional', 'rank')
REPLACEMENTS = ('generational', 'moran', 'death-birth', 'truncation', 'best-mutant')
CHANGES = ('other', 'any', 'one', 'step', 'new')
PARENTS = ('survivors', 'all')
MIN_ALPHABET = 2
HALF = 0.5


class EvolutionInteraction(Interaction):
    """An interaction that breeds the strategies of a group by selection and mutation.

    Every `period` steps the interaction selects parents by their fitness and replaces strategies by offspring. The
    `selection` picks parents by tournaments, by roulette, in proportion to fitness, or by rank. The `replacement` says
    how offspring take their places:

    - `'generational'`: every member is replaced by an offspring of a pair of parents, which exchange the tails of their
      strategies with probability `crossover`, and the `elite` fittest pass unchanged.
    - `'moran'`: one birth and death at a time, a selected parent copying onto a random member.
    - `'death-birth'`: members die with probability `death` and the offspring of survivors fill their places.
    - `'truncation'`: the `fraction` least fit are replaced by the offspring of the others.
    - `'best-mutant'`: every member draws a mutant, and the fittest mutant spreads when it beats the fittest strategy.

    Each gene of an offspring mutates with probability `mutation` by the rule `change`, or a function
    `mutation(strategy, rng)` returns the mutated copy. The fitness is a state of the members, or the function
    `evaluate(strategies, system)`, which the replacement writes into the fitness state.

    Unlike `ImitationInteraction`, by which members copy neighbours, it replaces strategies by the offspring of the
    fittest across the group. Choose it when strategies evolve by selection, crossover, and mutation.

    It appears in many fields. Genetic algorithms search for designs, strategies of the iterated prisoner's dilemma
    evolve in tournaments, and genotypes evolve on a fitness landscape in population genetics.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the agent group whose strategies evolve.

        strategy:
            The name of the state of the members that holds the strategy, a sequence of genes, or the names of
            several states, one gene each.

        fitness:
            The name of the state of the members that holds the fitness, or `None` for a neutral population of equal
            fitness when no `evaluate` is given.

        selection:
            The selection of parents, `'tournament'`, `'roulette'`, `'proportional'`, or `'rank'`.

        replacement:
            The replacement of strategies, `'generational'`, `'moran'`, `'death-birth'`, `'truncation'`, or
            `'best-mutant'`.

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

        change:
            How a gene mutates, `'other'`, `'any'`, `'one'`, `'step'`, or `'new'`.

        parents:
            The members the death-birth replacement selects parents among, `'survivors'` or `'all'`.

        fraction:
            The fraction of least fit members that the truncation replacement replaces.

        period:
            The number of steps, or rounds of the phase, between two replacements.

        evaluate:
            The function `evaluate(strategies, system)` that returns the fitness of every row of strategies, or `None`
            to read the fitness state.

        reset:
            The values of states that every replaced member takes, or `None` for no reset.

        phase:
            The phase of the step the generation runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            A variant is unknown, a probability is outside `[0, 1]`, the mutation is neither a probability nor a
            function, a count is invalid, or the evaluation is not callable.

    Examples:
        >>> from types import SimpleNamespace
        >>> from skcomplex.systems.general.agents import EvolutionInteraction
        >>> evolution = EvolutionInteraction('evolution', target='agents', selection='roulette', phase='breeding')
        >>> evolution.prepare().run(SimpleNamespace()).laws_[0].phase
        'breeding'
        >>> EvolutionInteraction('evolution', target='agents', selection='rank', replacement='moran').selection
        'rank'
        >>> EvolutionInteraction(
        ...     'evolution', target='agents', replacement='truncation', fraction=0.2, change='any', period=50
        ... ).period
        50
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[
            str,
            RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP), states=('strategy',)),
        ],
        strategy: str | Sequence[str] = 'strategy',
        fitness: str | None = 'fitness',
        selection: str = 'tournament',
        replacement: str = 'generational',
        crossover: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.7,
        mutation: float | Callable[[Array, np.random.Generator], Array] = 0.01,
        elite: int = 0,
        tournament: int = 2,
        alphabet: int | None = 2,
        scale: float = 0.1,
        death: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        change: str = 'other',
        parents: str = 'survivors',
        fraction: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        period: int = 1,
        evaluate: Callable[[Array, SystemViewProtocol], Array] | None = None,
        reset: Mapping[str, float] | None = None,
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
        self.change = change
        self.parents = parents
        self.fraction = fraction
        self.period = period
        self.evaluate = evaluate
        self.reset = reset
        self.phase = phase

    def _check_counts(self) -> None:
        """Raise an error for invalid counts of the evolution.

        It raises it for a negative elite, a tournament without entrants, an alphabet of one, or no period.
        """
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
        if self.period < 1:
            error_msg = f'{self.label!r} has the period {self.period}. Give a period of at least 1.'
            raise ValidationError(error_msg)

    def _check_probabilities(self) -> None:
        """Raise an error for an invalid probability or mutation.

        A probability is outside `[0, 1]`, or a mutation is neither a probability nor callable.
        """
        probabilities = [('crossover', self.crossover), ('death', self.death), ('fraction', self.fraction)]
        if not callable(self.mutation):
            probabilities.append(('mutation', self.mutation))
        for name, value in probabilities:
            if not isinstance(value, int | float) or not 0 <= value <= 1:
                error_msg = f'The {name} probability of {self.label!r} is {value!r}. Give a value in [0, 1].'
                raise ValidationError(error_msg)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown variant, a bad probability, or an invalid count.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, a variant is unknown, a probability is outside `[0, 1]`, the mutation is
                neither a probability nor a function, a count is invalid, or the evaluation is not callable.
        """
        super()._prepare(context)
        self._check_variant('selection', self.selection, SELECTIONS, (BaseSpec,))
        self._check_variant('replacement', self.replacement, REPLACEMENTS, (BaseSpec,))
        for name, value, variants in (
            ('change', self.change, CHANGES),
            ('parents', self.parents, PARENTS),
        ):
            self._check_variant(name, value, variants)
        if (isinstance(self.strategy, str) and not self.strategy) or not isinstance(self.strategy, str | Sequence):
            error_msg = f'{self.label!r} has the strategy {self.strategy!r}. Give a state name or a sequence of them.'
            raise ValidationError(error_msg)
        if self.evaluate is not None and not callable(self.evaluate):
            error_msg = f'The evaluation of {self.label!r} must be callable, but it is {self.evaluate!r}.'
            raise ValidationError(error_msg)
        self._check_probabilities()
        self._check_counts()

    def _get_value(self, name: str, system: SystemViewProtocol) -> float:
        """Return a probability.

        It is the parameter of the system when the class declares it as one.
        """
        value = getattr(self, self._get_parameter(name))
        if self._declares(self._get_parameter(name)) or isinstance(value, int | float):
            return self._get_parameter_value(name, system)
        return 0.0

    def _evaluate(self, strategies: Array, system: SystemViewProtocol) -> Array | None:
        """Return the fitness of every row of strategies by the evaluation.

        It returns `None` when there is no evaluation.

        Args:
            strategies:
                The strategies, one row per member, in the shape the states hold them.

            system:
                The system view of the interaction.

        Returns:
            fitness:
                The fitness of every row, or `None` to read the fitness state.
        """
        if self.evaluate is None:
            return None
        return np.asarray(self.evaluate(strategies, system), dtype=float).reshape(len(strategies))

    def _evaluate_rows(self, flat: Array, genes: tuple[int, ...], system: SystemViewProtocol) -> Array | None:
        """Return the fitness of flat rows of genes.

        The rows get back the shape of the states before evaluation.
        """
        return self._evaluate(flat.reshape(len(flat), *genes), system)

    def _select(self, fitness: Array, count: int, rng: np.random.Generator) -> npt.NDArray[np.intp]:
        """Return the indices of a number of parents selected on their fitness."""
        size = len(fitness)
        if self.selection == 'tournament':
            entrants = rng.integers(size, size=(count, self.tournament))
            return entrants[np.arange(count), np.argmax(fitness[entrants], axis=1)]
        if self.selection == 'rank':
            weights = np.empty(size)
            weights[np.argsort(fitness, kind='stable')] = np.arange(1, size + 1)
        elif self.selection == 'proportional':
            weights = fitness.copy()
        elif self.selection != 'roulette':
            raise self._get_variant_error(self.selection, 'selection', '_select')
        else:
            weights = fitness - fitness.min()
        weights = weights if weights.sum() > 0 else np.ones_like(weights)
        return rng.choice(size, size=count, p=weights / weights.sum())

    def _get_cross(self, parents: Array, probability: float, rng: np.random.Generator) -> Array:
        """Return the parents after one-point crossover of every consecutive pair."""
        children = parents.copy()
        length = children.shape[1]
        for first in range(0, len(children) - 1, 2):
            if length > 1 and rng.random() < probability:
                cut = int(rng.integers(1, length))
                tail = children[first, cut:].copy()
                children[first, cut:] = children[first + 1, cut:]
                children[first + 1, cut:] = tail
        return children

    def _shift(self, children: Array, mutated: npt.NDArray[np.bool_], rng: np.random.Generator) -> Array:
        """Return the children whose mutated genes change to another symbol.

        The genes may also change by Gaussian noise.
        """
        if self.alphabet is None:
            return children + mutated * rng.normal(0.0, self.scale, children.shape)
        shift = rng.integers(1, self.alphabet, children.shape)
        return np.where(mutated, (children + shift) % self.alphabet, children)

    def _redraw(self, children: Array, mutated: npt.NDArray[np.bool_], rng: np.random.Generator) -> Array:
        """Return the children whose mutated genes are drawn anew from the alphabet.

        The genes may also change by Gaussian noise.
        """
        if self.alphabet is None:
            return children + mutated * rng.normal(0.0, self.scale, children.shape)
        return np.where(mutated, rng.integers(0, self.alphabet, children.shape), children)

    def _step(self, children: Array, mutated: npt.NDArray[np.bool_], rng: np.random.Generator) -> Array:
        """Return the children whose mutated genes move one step up or down.

        The genes stay within the alphabet when it is finite.
        """
        steps = np.where(rng.random(children.shape) < HALF, 1.0, -1.0)
        stepped = children + mutated * steps
        return stepped if self.alphabet is None else stepped % self.alphabet

    def _mutate(self, children: Array, probability: float, rng: np.random.Generator, population: Array) -> Array:
        """Return the children mutated by the rule of the change or the mutation function.

        Args:
            children:
                The strategies of the offspring, one row per offspring.

            probability:
                The probability of mutation.

            rng:
                The random number generator.

            population:
                The strategies the members hold, whose largest label the rule `'new'` exceeds.

        Returns:
            children:
                The mutated strategies.
        """
        if callable(self.mutation):
            mutation = self.mutation
            return np.array([np.ravel(mutation(child.copy(), rng)) for child in children], dtype=float)
        if self.change == 'one':
            mutated = np.zeros(children.shape, dtype=bool)
            chosen = rng.random(len(children)) < probability
            mutated[np.arange(len(children)), rng.integers(children.shape[1], size=len(children))] = chosen
            return self._shift(children, mutated, rng)
        mutated = rng.random(children.shape) < probability
        if self.change == 'any':
            return self._redraw(children, mutated, rng)
        if self.change == 'step':
            return self._step(children, mutated, rng)
        if self.change == 'new':
            labels = population.max(initial=0.0) + np.cumsum(mutated).reshape(children.shape)
            return np.where(mutated, labels, children)
        return self._shift(children, mutated, rng)

    def _breed_generation(self, flat: Array, fitness: Array, system: SystemViewProtocol) -> Array:
        """Return the strategies of the next generation, one row per member."""
        rng = system.rng
        parents = flat[self._select(fitness, len(flat), rng)]
        children = self._get_cross(parents, self._get_value('crossover', system), rng)
        children = self._mutate(children, self._get_value('mutation', system), rng, flat)
        if self.elite:
            best = np.argsort(-fitness, kind='stable')[: self.elite]
            children[: len(best)] = flat[best]
        return children

    def _breed_moran(
        self,
        flat: Array,
        fitness: Array,
        system: SystemViewProtocol,
        genes: tuple[int, ...],
    ) -> tuple[Array, Array]:
        """Return the strategies and the fitness after one birth-death event per member.

        The events happen one after the other.
        """
        rng, probability = system.rng, self._get_value('mutation', system)
        strategies, fitness = flat.copy(), fitness.copy()
        for _ in range(len(strategies)):
            parent = int(self._select(fitness, 1, rng)[0])
            dying = int(rng.integers(len(strategies)))
            child = self._mutate(strategies[parent : parent + 1], probability, rng, strategies)
            evaluated = self._evaluate_rows(child, genes, system)
            strategies[dying] = child[0]
            fitness[dying] = fitness[parent] if evaluated is None else evaluated[0]
        return strategies, fitness

    def _breed_death_birth(
        self,
        flat: Array,
        fitness: Array,
        system: SystemViewProtocol,
    ) -> tuple[Array, npt.NDArray[np.bool_]]:
        """Return the strategies after the offspring of the survivors fill the dead.

        It also returns the dead members.
        """
        rng = system.rng
        dead = rng.random(len(flat)) < self._get_value('death', system)
        if not dead.any():
            return flat, dead
        everyone = self.parents == 'all' or dead.all()
        survivors = np.arange(len(flat)) if everyone else np.flatnonzero(~dead)
        parents = survivors[self._select(fitness[survivors], int(dead.sum()), rng)]
        strategies = flat.copy()
        strategies[dead] = self._mutate(flat[parents], self._get_value('mutation', system), rng, flat)
        return strategies, dead

    def _breed_truncation(
        self,
        flat: Array,
        fitness: Array,
        system: SystemViewProtocol,
    ) -> tuple[Array, npt.NDArray[np.bool_]]:
        """Return the strategies after the offspring of the others fill the least fit.

        It also returns the replaced members.
        """
        count = round(self._get_value('fraction', system) * len(flat))
        replaced = np.zeros(len(flat), dtype=bool)
        replaced[np.argsort(fitness, kind='stable')[:count]] = True
        if not count:
            return flat, replaced
        others = np.arange(len(flat)) if replaced.all() else np.flatnonzero(~replaced)
        parents = others[self._select(fitness[others], count, system.rng)]
        strategies = flat.copy()
        strategies[replaced] = self._mutate(flat[parents], self._get_value('mutation', system), system.rng, flat)
        return strategies, replaced

    def _breed_best_mutant(
        self,
        flat: Array,
        fitness: Array,
        system: SystemViewProtocol,
        genes: tuple[int, ...],
    ) -> Array:
        """Return the strategies after every member takes the fittest mutant.

        A member takes it when it is fitter or all mutated.
        """
        mutants = self._mutate(flat, self._get_value('mutation', system), system.rng, flat)
        mutated = np.any(mutants != flat, axis=1)
        if not mutated.any():
            return flat
        evaluated = self._evaluate_rows(mutants[mutated], genes, system)
        if evaluated is None:
            error_msg = f'{self.label!r} replaces by the best mutant, which needs an evaluation of the strategies.'
            raise SimulationError(error_msg)
        best = int(np.argmax(evaluated))
        if evaluated[best] <= fitness.max() and not mutated.all():
            return flat
        return np.repeat(mutants[mutated][best : best + 1], len(flat), axis=0)

    def _breed(
        self,
        strategies: Array,
        fitness: Array,
        system: SystemViewProtocol,
    ) -> tuple[Array, Array | None, npt.NDArray[np.bool_]]:
        """Return the strategies after the replacement, and the replaced members.

        It also returns their evaluated fitness, or `None`.
        """
        flat, genes = strategies.reshape(len(strategies), -1), strategies.shape[1:]
        replaced = np.ones(len(flat), dtype=bool)
        if self.replacement == 'moran':
            children, fitness = self._breed_moran(flat, fitness, system, genes)
        elif self.replacement == 'death-birth':
            children, replaced = self._breed_death_birth(flat, fitness, system)
        elif self.replacement == 'truncation':
            children, replaced = self._breed_truncation(flat, fitness, system)
        elif self.replacement == 'best-mutant':
            children = self._breed_best_mutant(flat, fitness, system, genes)
        elif self.replacement != 'generational':
            raise self._get_variant_error(self.replacement, 'replacement', '_breed')
        else:
            children = self._breed_generation(flat, fitness, system)
        children = children.reshape(strategies.shape)
        return children, self._evaluate(children, system), replaced

    def _read(self, agents: Sequence[AgentViewProtocol], system: SystemViewProtocol) -> tuple[Array, Array]:
        """Return the strategies, one row per member, and their fitness."""
        if isinstance(self.strategy, str):
            strategies = np.array([getattr(agent, self.strategy) for agent in agents], dtype=float)
        else:
            strategies = np.array([[getattr(agent, name) for name in self.strategy] for agent in agents], dtype=float)
        evaluated = self._evaluate(strategies, system)
        if evaluated is not None:
            return strategies, evaluated
        if self.fitness is None:
            return strategies, np.ones(len(agents))
        return strategies, np.array([getattr(agent, self.fitness) for agent in agents], dtype=float)

    def _write(
        self,
        agents: Sequence[AgentViewProtocol],
        children: Array,
        fitness: Array | None,
        replaced: npt.NDArray[np.bool_],
    ) -> None:
        """Write the strategies, the evaluated fitness, and the reset states.

        The reset states are those of the replaced members.
        """
        names = [self.strategy] if isinstance(self.strategy, str) else list(self.strategy)
        for index, (agent, child, new) in enumerate(zip(agents, children, replaced, strict=True)):
            genes = [child] if isinstance(self.strategy, str) else list(child)
            for name, gene in zip(names, genes, strict=True):
                setattr(agent, name, gene if np.ndim(gene) else float(gene))
            if fitness is not None and self.fitness is not None:
                setattr(agent, self.fitness, float(fitness[index]))
            for name, value in (self.reset or {}).items() if new else ():
                setattr(agent, name, value)

    def _evolve(self, system: SystemViewProtocol) -> None:
        """Replace the strategies of the members by their offspring, once every period."""
        calls = system.storage.get('calls', 0)
        calls = calls + 1 if isinstance(calls, int) else 1
        system.storage['calls'] = calls
        agents = system.get_agents(self._get_role('target'))
        if not agents or calls % self.period:
            return
        strategies, fitness = self._read(agents, system)
        self._write(agents, *self._breed(strategies, fitness, system))

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
