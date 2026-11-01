"""Define evolution interactions, which breed the strategies of a group by selection, crossover, and mutation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.sets.real_set import RealSet

from ...errors import SimulationError, ValidationError
from ...specs import ParameterSpec, RoleSpec
from ...types import Array
from ..laws import BaseLaw, RuleLaw
from ..types import AgentViewProtocol, ModelViewProtocol
from ._interaction import Interaction

SELECTIONS = ('tournament', 'roulette', 'proportional', 'rank')
REPLACEMENTS = ('generational', 'moran', 'death-birth', 'truncation', 'best-mutant')
CHANGES = ('other', 'any', 'one', 'step', 'new')
PARENTS = ('survivors', 'all')
MIN_ALPHABET = 2
HALF = 0.5


class EvolutionInteraction(Interaction):
    """Replacement of the strategies of an agent group by offspring selected on a fitness, with crossover and mutation.

    Every `period` steps, or rounds of its phase, the interaction selects parents by tournaments of `tournament`
    entrants, by roulette with weights of the fitness minus its minimum, in proportion to the fitness itself
    (`'proportional'`), or by rank with weights `1` for the least fit to `n` for the fittest of `n` members. The
    `'generational'` replacement selects as many parents as there are members, each pair of which exchanges the tails
    after a random cut with probability `crossover`, and passes the `elite` fittest strategies unchanged. The `'moran'`
    replacement runs as many birth-death events as there are members, one after the other, each copying a selected
    parent onto a member drawn uniformly, and the copy takes the fitness of its parent for the later events of the
    step. The `'death-birth'` replacement removes every member with probability `death` and fills its place with the
    offspring of a parent selected among the survivors, or among all members when none survives or `parents` is
    `'all'`. The `'truncation'` replacement fills the places of the `fraction` least fit members with the offspring of
    parents selected among the others. The `'best-mutant'` replacement lets every member draw a mutant of its strategy,
    and every member takes the fittest mutant when it is fitter than the fittest strategy, or when every member
    mutated, so a population of one strategy climbs a landscape as a whole.

    Offspring mutate: each gene with probability `mutation` changes by the rule `change`, to another symbol of an
    alphabet of `alphabet` symbols (`'other'`), to a symbol drawn anew from the alphabet (`'any'`), one step up or down
    (`'step'`), or to a label one above the largest the members hold (`'new'`), or with probability `mutation` one
    gene drawn at random of every offspring changes to another symbol (`'one'`). When `alphabet` is `None`, the
    `'other'` and `'any'` rules add Gaussian noise of deviation `scale`. A function `mutation(strategy, rng)` returns
    the mutated copy of a strategy instead. The fitness is a state of the members, or the function
    `evaluate(strategies, model)` of the strategies, one row per member, which the replacement then writes into the
    fitness state. The probabilities are parameters of the model when a subclass declares them with a
    `ParameterSpec`, as the base does for `crossover`, `death`, and `fraction`. The draws come from the random stream
    of the model.

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
            The function `evaluate(strategies, model)` that returns the fitness of every row of strategies, or `None`
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
        >>> from skcomplex.core.interactions import EvolutionInteraction
        >>> evolution = EvolutionInteraction('evolution', target='agents', selection='roulette', phase='breeding')
        >>> evolution.check().prepare().run(SimpleNamespace()).laws_[0].phase
        'breeding'
        >>> EvolutionInteraction('evolution', target='agents', selection='rank', replacement='moran').check().selection
        'rank'
        >>> EvolutionInteraction(
        ...     'evolution', target='agents', replacement='truncation', fraction=0.2, change='any', period=50
        ... ).check().period
        50
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
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
        evaluate: Callable[[Array, ModelViewProtocol], Array] | None = None,
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
        """Raise an error for a negative elite, a tournament without entrants, an alphabet of one, or no period."""
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
        """Raise an error for a probability outside `[0, 1]` or a mutation neither a probability nor callable."""
        probabilities = [('crossover', self.crossover), ('death', self.death), ('fraction', self.fraction)]
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
                The type declares states, a variant is unknown, a probability is outside `[0, 1]`, the mutation is
                neither a probability nor a function, a count is invalid, or the evaluation is not callable.
        """
        super()._check()
        for name, value, variants in (
            ('selection', self.selection, SELECTIONS),
            ('replacement', self.replacement, REPLACEMENTS),
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

    def _value(self, name: str, model: ModelViewProtocol) -> float:
        """Return a probability, the parameter of the model when the class declares it as one."""
        value = getattr(self, self._get_parameter(name))
        if self._declares(self._get_parameter(name)) or isinstance(value, int | float):
            return self._get_parameter_value(name, model)
        return 0.0

    def _evaluate(self, strategies: Array, model: ModelViewProtocol) -> Array | None:
        """Return the fitness of every row of strategies by the evaluation, or `None` when there is none.

        Args:
            strategies:
                The strategies, one row per member, in the shape the states hold them.

            model:
                The model view of the interaction.

        Returns:
            fitness:
                The fitness of every row, or `None` to read the fitness state.
        """
        if self.evaluate is None:
            return None
        return np.asarray(self.evaluate(strategies, model), dtype=float).reshape(len(strategies))

    def _evaluate_rows(self, flat: Array, genes: tuple[int, ...], model: ModelViewProtocol) -> Array | None:
        """Return the fitness of flat rows of genes, given back the shape of the states before evaluation."""
        return self._evaluate(flat.reshape(len(flat), *genes), model)

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
        else:
            weights = fitness - fitness.min()
        weights = weights if weights.sum() > 0 else np.ones_like(weights)
        return rng.choice(size, size=count, p=weights / weights.sum())

    def _cross(self, parents: Array, probability: float, rng: np.random.Generator) -> Array:
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
        """Return the children whose mutated genes change to another symbol, or by Gaussian noise."""
        if self.alphabet is None:
            return children + mutated * rng.normal(0.0, self.scale, children.shape)
        shift = rng.integers(1, self.alphabet, children.shape)
        return np.where(mutated, (children + shift) % self.alphabet, children)

    def _redraw(self, children: Array, mutated: npt.NDArray[np.bool_], rng: np.random.Generator) -> Array:
        """Return the children whose mutated genes are drawn anew from the alphabet, or by Gaussian noise."""
        if self.alphabet is None:
            return children + mutated * rng.normal(0.0, self.scale, children.shape)
        return np.where(mutated, rng.integers(0, self.alphabet, children.shape), children)

    def _step(self, children: Array, mutated: npt.NDArray[np.bool_], rng: np.random.Generator) -> Array:
        """Return the children whose mutated genes move one step up or down, within the alphabet when it is finite."""
        steps = np.where(rng.random(children.shape) < HALF, 1.0, -1.0)
        stepped = children + mutated * steps
        return stepped if self.alphabet is None else stepped % self.alphabet

    def _mutate(self, children: Array, probability: float, rng: np.random.Generator, population: Array) -> Array:
        """Return the children mutated by the rule of the change, or by the mutation function.

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

    def _breed_generation(self, flat: Array, fitness: Array, model: ModelViewProtocol) -> Array:
        """Return the strategies of the next generation, one row per member."""
        rng = model.rng
        parents = flat[self._select(fitness, len(flat), rng)]
        children = self._cross(parents, self._value('crossover', model), rng)
        children = self._mutate(children, self._value('mutation', model), rng, flat)
        if self.elite:
            best = np.argsort(-fitness, kind='stable')[: self.elite]
            children[: len(best)] = flat[best]
        return children

    def _breed_moran(
        self,
        flat: Array,
        fitness: Array,
        model: ModelViewProtocol,
        genes: tuple[int, ...],
    ) -> tuple[Array, Array]:
        """Return the strategies and the fitness after one birth-death event per member, one after the other."""
        rng, probability = model.rng, self._value('mutation', model)
        strategies, fitness = flat.copy(), fitness.copy()
        for _ in range(len(strategies)):
            parent = int(self._select(fitness, 1, rng)[0])
            dying = int(rng.integers(len(strategies)))
            child = self._mutate(strategies[parent : parent + 1], probability, rng, strategies)
            evaluated = self._evaluate_rows(child, genes, model)
            strategies[dying] = child[0]
            fitness[dying] = fitness[parent] if evaluated is None else evaluated[0]
        return strategies, fitness

    def _breed_death_birth(
        self,
        flat: Array,
        fitness: Array,
        model: ModelViewProtocol,
    ) -> tuple[Array, npt.NDArray[np.bool_]]:
        """Return the strategies after the dead members are filled by the offspring of the survivors, and the dead."""
        rng = model.rng
        dead = rng.random(len(flat)) < self._value('death', model)
        if not dead.any():
            return flat, dead
        everyone = self.parents == 'all' or dead.all()
        survivors = np.arange(len(flat)) if everyone else np.flatnonzero(~dead)
        parents = survivors[self._select(fitness[survivors], int(dead.sum()), rng)]
        strategies = flat.copy()
        strategies[dead] = self._mutate(flat[parents], self._value('mutation', model), rng, flat)
        return strategies, dead

    def _breed_truncation(
        self,
        flat: Array,
        fitness: Array,
        model: ModelViewProtocol,
    ) -> tuple[Array, npt.NDArray[np.bool_]]:
        """Return the strategies after the least fit are filled by the offspring of the others, and the replaced."""
        count = round(self._value('fraction', model) * len(flat))
        replaced = np.zeros(len(flat), dtype=bool)
        replaced[np.argsort(fitness, kind='stable')[:count]] = True
        if not count:
            return flat, replaced
        others = np.arange(len(flat)) if replaced.all() else np.flatnonzero(~replaced)
        parents = others[self._select(fitness[others], count, model.rng)]
        strategies = flat.copy()
        strategies[replaced] = self._mutate(flat[parents], self._value('mutation', model), model.rng, flat)
        return strategies, replaced

    def _breed_best_mutant(
        self,
        flat: Array,
        fitness: Array,
        model: ModelViewProtocol,
        genes: tuple[int, ...],
    ) -> Array:
        """Return the strategies after every member takes the fittest mutant, when it is fitter or all mutated."""
        mutants = self._mutate(flat, self._value('mutation', model), model.rng, flat)
        mutated = np.any(mutants != flat, axis=1)
        if not mutated.any():
            return flat
        evaluated = self._evaluate_rows(mutants[mutated], genes, model)
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
        model: ModelViewProtocol,
    ) -> tuple[Array, Array | None, npt.NDArray[np.bool_]]:
        """Return the strategies after the replacement, their evaluated fitness or `None`, and the replaced."""
        flat, genes = strategies.reshape(len(strategies), -1), strategies.shape[1:]
        replaced = np.ones(len(flat), dtype=bool)
        if self.replacement == 'moran':
            children, fitness = self._breed_moran(flat, fitness, model, genes)
        elif self.replacement == 'death-birth':
            children, replaced = self._breed_death_birth(flat, fitness, model)
        elif self.replacement == 'truncation':
            children, replaced = self._breed_truncation(flat, fitness, model)
        elif self.replacement == 'best-mutant':
            children = self._breed_best_mutant(flat, fitness, model, genes)
        else:
            children = self._breed_generation(flat, fitness, model)
        children = children.reshape(strategies.shape)
        return children, self._evaluate(children, model), replaced

    def _read(self, agents: Sequence[AgentViewProtocol], model: ModelViewProtocol) -> tuple[Array, Array]:
        """Return the strategies, one row per member, and their fitness."""
        if isinstance(self.strategy, str):
            strategies = np.array([getattr(agent, self.strategy) for agent in agents], dtype=float)
        else:
            strategies = np.array([[getattr(agent, name) for name in self.strategy] for agent in agents], dtype=float)
        evaluated = self._evaluate(strategies, model)
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
        """Write the strategies, the evaluated fitness, and the reset states of the replaced members."""
        names = [self.strategy] if isinstance(self.strategy, str) else list(self.strategy)
        for index, (agent, child, new) in enumerate(zip(agents, children, replaced, strict=True)):
            genes = [child] if isinstance(self.strategy, str) else list(child)
            for name, gene in zip(names, genes, strict=True):
                setattr(agent, name, gene if np.ndim(gene) else float(gene))
            if fitness is not None and self.fitness is not None:
                setattr(agent, self.fitness, float(fitness[index]))
            for name, value in (self.reset or {}).items() if new else ():
                setattr(agent, name, value)

    def _evolve(self, model: ModelViewProtocol) -> None:
        """Replace the strategies of the members by their offspring, once every period."""
        calls = model.storage.get('calls', 0)
        calls = calls + 1 if isinstance(calls, int) else 1
        model.storage['calls'] = calls
        agents = model.agents(self._get_role('target'))
        if not agents or calls % self.period:
            return
        strategies, fitness = self._read(agents, model)
        self._write(agents, *self._breed(strategies, fitness, model))

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
