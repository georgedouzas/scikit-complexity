"""Search processes on a landscape: population optimisers and simulated annealing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from sage.all__sagemath_symbolics import Expression, cos, pi

from ....core.components.base import BasePart
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction, SamplingInteraction
from ....core.components.laws import BaseLaw, RuleLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BallNeighborhood, BaseSpace, DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.components.types import SystemViewProtocol
from ....core.errors import ValidationError
from ....core.specs import BaseSpec, ObservableSpec, RoleSpec, StateSpec
from ....core.types import Array, ElementKind


@dataclass(frozen=True)
class DifferentialEvolutionSpec(BaseSpec):
    """Settings of differential evolution.

    Args:
        variant:
            The scheme of pymoo's differential evolution, such as `'DE/best/1/bin'`, its default.

        weight:
            The differential weight of the mutation, by default that of pymoo.

        crossover:
            The crossover probability, by default that of pymoo.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import DifferentialEvolutionSpec
        >>> DifferentialEvolutionSpec().variant
        'DE/best/1/bin'
    """

    variant: str = 'DE/best/1/bin'
    weight: float = 0.5
    crossover: float = 0.2


@dataclass(frozen=True)
class ParticleSwarmSpec(BaseSpec):
    """Settings of particle swarm optimisation.

    Args:
        inertia:
            The weight of the previous velocity.

        cognitive:
            The attraction to the best position of a particle.

        social:
            The attraction to the best position of the swarm.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import ParticleSwarmSpec
        >>> ParticleSwarmSpec().inertia
        0.9
    """

    inertia: float = 0.9
    cognitive: float = 2.0
    social: float = 2.0


@dataclass(frozen=True)
class EvolutionStrategySpec(BaseSpec):
    """Settings of an evolution strategy.

    Args:
        offspring:
            The number of offspring per generation, or `None` for pymoo's default.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import EvolutionStrategySpec
        >>> EvolutionStrategySpec(offspring=200).offspring
        200
    """

    offspring: int | None = None


@dataclass(frozen=True)
class CovarianceAdaptationSpec(BaseSpec):
    """Settings of covariance matrix adaptation.

    Args:
        sigma:
            The initial step size, relative to the box of the candidates.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import CovarianceAdaptationSpec
        >>> CovarianceAdaptationSpec().sigma
        0.1
    """

    sigma: float = 0.1


type AlgorithmSpec = DifferentialEvolutionSpec | ParticleSwarmSpec | EvolutionStrategySpec | CovarianceAdaptationSpec

ALGORITHM_SPECS = (DifferentialEvolutionSpec, ParticleSwarmSpec, EvolutionStrategySpec, CovarianceAdaptationSpec)
SEEDS = 2**31


class OptimizationInteraction(Interaction):
    """Move a group of candidates on an objective.

    It runs one generation of a pymoo algorithm per step of discrete time. Every candidate holds a vector state in a
    box, from which the first generation starts. Every step asks the algorithm for candidates, evaluates the objective
    on their array, tells the algorithm the values, and writes the candidates of the generation into the members. The
    interaction reports `best`, the lowest value so far, and `mean`, the mean value of the generation, every step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of candidates.

        state:
            The name of the vector state of the candidates.

        objective:
            The function of the array of candidates, one row per candidate, that returns their values.

        bounds:
            The lower and the upper bounds of the box, each a number or one number per component.

        algorithm:
            The settings of the algorithm, which select it.

        phase:
            The name of the phase of the step the rule runs in, or `None` for the default phase.

    Raises:
        ValidationError:
            The algorithm is not an algorithm spec, or the objective is not callable, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import OptimizationInteraction
        >>> objective = lambda values: (values**2).sum(axis=1)
        >>> search = OptimizationInteraction('search', target='swarm', state='x', objective=objective, bounds=(-1, 1))
        >>> search.label
        'search'
    """

    reports: ClassVar[tuple[str, ...]] = ('best', 'mean')

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.GROUP, ElementKind.VARIABLE_GROUP), states=('state',))],
        state: str,
        objective: Callable[[Array], npt.ArrayLike],
        bounds: tuple[npt.ArrayLike, npt.ArrayLike],
        algorithm: AlgorithmSpec = DifferentialEvolutionSpec(),
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.objective = objective
        self.bounds = bounds
        self.algorithm = algorithm
        self.phase = phase

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an algorithm that is not a spec, or an uncallable objective.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The algorithm is not an algorithm spec, or the objective is not callable.
        """
        super()._prepare(context)
        if not isinstance(self.algorithm, ALGORITHM_SPECS):
            names = ', '.join(item.__name__ for item in ALGORITHM_SPECS)
            error_msg = f'The algorithm {self.algorithm!r} of {self.label!r} is no algorithm spec. Use one of {names}.'
            raise ValidationError(error_msg)
        if not callable(self.objective):
            error_msg = f'The objective of {self.label!r} must be a function of the array of candidates.'
            raise ValidationError(error_msg)

    def _differential_evolution(self, spec: DifferentialEvolutionSpec, candidates: Array) -> Algorithm:
        """Return differential evolution from the candidates."""
        return DE(
            pop_size=len(candidates),
            sampling=candidates,
            variant=spec.variant,
            F=spec.weight,
            CR=spec.crossover,
        )

    def _particle_swarm(self, spec: ParticleSwarmSpec, candidates: Array) -> Algorithm:
        """Return particle swarm optimisation from the candidates."""
        return PSO(
            pop_size=len(candidates),
            sampling=candidates,
            w=spec.inertia,
            c1=spec.cognitive,
            c2=spec.social,
            adaptive=False,
        )

    def _evolution_strategy(self, spec: EvolutionStrategySpec, candidates: Array) -> Algorithm:
        """Return an evolution strategy from the candidates."""
        return ES(pop_size=len(candidates), n_offsprings=spec.offspring, sampling=candidates)

    def _covariance_adaptation(self, spec: CovarianceAdaptationSpec, candidates: Array) -> Algorithm:
        """Return covariance matrix adaptation from the mean of the candidates."""
        return CMAES(x0=candidates.mean(axis=0), sigma=spec.sigma, pop_size=len(candidates))

    def _get_algorithm(self, candidates: Array, seed: int) -> Algorithm:
        """Return the algorithm of the spec.

        It is set up on the box of the candidates with a seed.
        """
        makers: dict[type, Callable[..., Algorithm]] = {
            DifferentialEvolutionSpec: self._differential_evolution,
            ParticleSwarmSpec: self._particle_swarm,
            EvolutionStrategySpec: self._evolution_strategy,
            CovarianceAdaptationSpec: self._covariance_adaptation,
        }
        size = candidates.shape[1]
        lower, upper = (np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in self.bounds)
        problem = Problem(n_var=size, n_obj=1, xl=lower, xu=upper)
        algorithm = makers[type(self.algorithm)](self.algorithm, candidates)
        algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
        return algorithm

    @staticmethod
    def _get_members_of(generation: Array, size: int) -> Array:
        """Return one row per member from the candidates of a generation.

        It repeats them when they are fewer.
        """
        return generation[np.arange(size) % len(generation)]

    def _write(self, agents: Sequence[object], candidates: Array) -> None:
        """Write the candidates into the members."""
        for agent, row in zip(agents, self._get_members_of(candidates, len(agents)), strict=True):
            setattr(agent, self.state, row)

    def _advance(self, system: SystemViewProtocol) -> None:
        """Advance the population by one generation and report its values.

        It does not advance once the algorithm has converged.
        """
        agents = system.get_agents(self._get_role('target'))
        if 'algorithm' not in system.storage:
            candidates = np.array([getattr(agent, self.state) for agent in agents], dtype=float)
            seed = int(system.rng.integers(SEEDS))
            system.storage.update(algorithm=self._get_algorithm(candidates, seed), best=np.inf)
        algorithm = cast(Algorithm, system.storage['algorithm'])
        if not algorithm.has_next():
            system.report('best', cast(float, system.storage['best']))
            system.report('mean', cast(float, system.storage['mean']))
            return
        population = algorithm.ask()
        generation = population.get('X')
        values = np.asarray(self.objective(generation), dtype=float).ravel()
        population.set('F', values[:, None])
        algorithm.tell(infills=population)
        best = min(cast(float, system.storage['best']), float(values.min()))
        system.storage.update(best=best, mean=float(values.mean()))
        self._write(agents, generation)
        system.report('best', best)
        system.report('mean', float(values.mean()))

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of one generation.

        Args:
            namespace:
                The namespace of the symbols the laws can use, which the rule does not read.

        Returns:
            laws:
                The rule that advances the population.
        """
        return [RuleLaw(self._advance, phase=self.phase)]


class AnnealedCandidateElement(Element):
    """Candidate solution of the plane.

    Its coordinates are sampled one state at a time.

    Args:
        label:
            The label of the candidate or the group.

        x:
            The first coordinate.

        y:
            The second coordinate.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import AnnealedCandidateElement
        >>> AnnealedCandidateElement('candidate', x=1.0).x
        1.0
    """

    def __init__(
        self,
        label: str,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y


class RunningBestElement(Element):
    """Lowest objective value of a search so far.

    Args:
        label:
            The label of the element.

        best:
            The value before the search starts.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import RunningBestElement
        >>> RunningBestElement('record', best=5.0).best
        5.0
    """

    def __init__(self, label: str, best: Annotated[float, StateSpec()] = 1e300) -> None:
        super().__init__(label)
        self.best = best


class SearchCandidateElement(Element):
    """Candidate solution that holds a vector.

    Args:
        label:
            The label of the candidate or the group.

        x:
            The vector of the candidate.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import SearchCandidateElement
        >>> SearchCandidateElement('candidate', x=(0.0, 1.0)).x
        (0.0, 1.0)
    """

    def __init__(self, label: str, x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = (0.0, 0.0)) -> None:
        super().__init__(label)
        self.x = x


class RunningBestInteraction(Interaction):
    """Keep the lowest value of an observable.

    It sets `best <- value if value < best else best`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element that holds `best`.

        source:
            The path of the observable.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import RunningBestInteraction
        >>> RunningBestInteraction('update', target='record', source='search/lowest').source
        'search/lowest'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BasePart]:
        """Return the update of the lowest value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update of `best`.
        """
        return [UpdateLaw(s.target.best, where(s.source.value < s.target.best, s.source.value, s.target.best))]


ALGORITHMS: dict[
    str,
    DifferentialEvolutionSpec | ParticleSwarmSpec | EvolutionStrategySpec | CovarianceAdaptationSpec,
] = {
    'differential-evolution': DifferentialEvolutionSpec(),
    'particle-swarm': ParticleSwarmSpec(),
    'evolution-strategy': EvolutionStrategySpec(),
    'covariance-adaptation': CovarianceAdaptationSpec(),
}
ANNEALING = 'annealing'
COOLED = 1e-3
N_DIM = 2


def _rastrigin(x: Expression, y: Expression) -> Expression:
    """Return the two-dimensional Rastrigin function at symbols."""
    return 20 + x**2 - 10 * cos(2 * pi * x) + y**2 - 10 * cos(2 * pi * y)


def _rosenbrock(x: Expression, y: Expression) -> Expression:
    """Return the two-dimensional Rosenbrock function at symbols."""
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def _rastrigin_values(values: Array) -> Array:
    """Return the Rastrigin function of every candidate."""
    return 20.0 + (values**2 - 10.0 * np.cos(2.0 * np.pi * values)).sum(axis=1)


def _rosenbrock_values(values: Array) -> Array:
    """Return the Rosenbrock function of every candidate."""
    return (1.0 - values[:, 0]) ** 2 + 100.0 * (values[:, 1] - values[:, 0] ** 2) ** 2


def _rastrigin_energy(s: SimpleNamespace) -> Expression:
    """Return the Rastrigin function of a candidate of a sampling."""
    return _rastrigin(s.target.x, s.target.y)


def _rosenbrock_energy(s: SimpleNamespace) -> Expression:
    """Return the Rosenbrock function of a candidate of a sampling."""
    return _rosenbrock(s.target.x, s.target.y)


def _rastrigin_member(member: SimpleNamespace) -> Expression:
    """Return the Rastrigin function of a member, for an observable."""
    return _rastrigin(member.x, member.y)


def _rosenbrock_member(member: SimpleNamespace) -> Expression:
    """Return the Rosenbrock function of a member, for an observable."""
    return _rosenbrock(member.x, member.y)


LANDSCAPES: dict[str, tuple[Callable[[SimpleNamespace], Expression], Callable[[Array], Array], float]] = {
    'rastrigin': (_rastrigin_energy, _rastrigin_values, 5.12),
    'rosenbrock': (_rosenbrock_energy, _rosenbrock_values, 2.048),
}
MEMBER_ENERGIES = {'rastrigin': _rastrigin_member, 'rosenbrock': _rosenbrock_member}


class EvolutionaryOptimisersSystem(BaseSystem):
    """A population of candidates that searches a two-dimensional landscape.

    Each step is one generation. A population optimiser moves the candidates through `OptimizationInteraction`, and
    reports `search.best`. Simulated annealing samples every coordinate through `SamplingInteraction` at a temperature
    that falls geometrically to 1e-3 of its start over `generations` steps, and `tracker/record.best` holds the lowest
    energy it has reached, from the observable `search/lowest` that the catalog lists, so a run passes it in
    `observables` (Raman 2021 §6.3.3.3, §6.3.4). The system has the group `candidates`.

    Args:
        n_candidates:
            The number of candidates.

        algorithm:
            `'differential-evolution'`, `'particle-swarm'`, `'evolution-strategy'`, `'covariance-adaptation'`, or
            `'annealing'`.

        landscape:
            `'rastrigin'` or `'rosenbrock'`.

        temperature:
            The initial temperature of annealing.

        width:
            The standard deviation of a move of annealing.

        generations:
            The number of steps over which annealing cools.

        random_state:
            The seed or generator of the initial candidates.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The algorithm or the landscape is unknown.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import EvolutionaryOptimisersSystem
        >>> EvolutionaryOptimisersSystem(10).prepare().system_.interactions[0].label
        'search'
    """

    field = 'computer_science/evolutionary_computation'
    topics = ('optimisation', 'evolutionary algorithm', 'annealing', 'fitness landscape')
    references = ('Raman 2021', 'Storn and Price 1997', 'Kirkpatrick, Gelatt, and Vecchi 1983')
    until = 100

    def __init__(
        self,
        n_candidates: int = 40,
        *,
        algorithm: str = 'differential-evolution',
        landscape: str = 'rastrigin',
        temperature: float = 5.0,
        width: float = 0.5,
        generations: float = 100.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_candidates = n_candidates
        self.algorithm = algorithm
        self.landscape = landscape
        self.temperature = temperature
        self.width = width
        self.generations = generations
        self.random_state = random_state

    @staticmethod
    def _annealing(
        start: Array,
        energy: Callable[[SimpleNamespace], Expression],
        temperature: float,
        width: float,
        generations: float,
        *,
        label: str,
    ) -> CompositeSystem:
        """Return candidates that anneal one state at a time, and the best-energy element.

        The element keeps the lowest energy the candidates reach.
        """
        candidates = GroupElement(
            'candidates',
            element=AnnealedCandidateElement,
            space=LatticeSpace((len(start),)),
            values={'x': start[:, 0], 'y': start[:, 1]},
            on='sites',
        )
        samplers = [
            SamplingInteraction(
                f'anneal_{state}',
                target='candidates',
                state=state,
                objective=energy,
                move='gaussian',
                width=width,
                intensity=1 / temperature,
                schedule='colored',
                neighborhood=BallNeighborhood(metric='manhattan', n_dim=1),
                cooling=lambda t: COOLED ** (t / generations),
            )
            for state in ('x', 'y')
        ]
        return CompositeSystem(
            label,
            systems=[
                CompositeSystem('search', elements=[candidates], interactions=samplers, time=DiscreteTime()),
                CompositeSystem('tracker', elements=[RunningBestElement('record')], time=DiscreteTime()),
            ],
            interactions=[RunningBestInteraction('update', target='tracker/record', source='search/lowest')],
            time=DiscreteTime(),
        )

    def _build_system(self) -> CompositeSystem:
        """Return a population of candidates that searches a two-dimensional landscape.

        Each step is one generation.
        """
        if (self.algorithm not in ALGORITHMS and self.algorithm != ANNEALING) or self.landscape not in LANDSCAPES:
            names = ', '.join((*ALGORITHMS, ANNEALING))
            error_msg = (
                f'EvolutionaryOptimisersSystem has algorithm={self.algorithm!r} and landscape={self.landscape!r}. '
                f'Use one of {names}, and one of {", ".join(LANDSCAPES)}.'
            )
            raise ValidationError(error_msg)
        energy, objective, bound = LANDSCAPES[self.landscape]
        start = np.random.default_rng(self.random_state).uniform(-bound, bound, (self.n_candidates, N_DIM))
        if self.algorithm == ANNEALING:
            return self._annealing(start, energy, self.temperature, self.width, self.generations, label=self.label)
        candidates = GroupElement(
            'candidates',
            element=SearchCandidateElement,
            size=self.n_candidates,
            values={'x': start},
        )
        search = OptimizationInteraction(
            'search',
            target='candidates',
            state='x',
            objective=objective,
            bounds=(-bound, bound),
            algorithm=ALGORITHMS[self.algorithm],
        )
        return CompositeSystem(self.label, elements=[candidates], interactions=[search], time=DiscreteTime())

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        They record the lowest candidate energy of simulated annealing. A population optimiser keeps the default
        settings. Simulated annealing nests the search and the tracker of the best energy, which run side by side, and
        records the lowest energy its candidates reach, as `search/lowest`, which the tracker reads.

        Returns:
            settings:
                The tuned run settings by name.

        Examples:
            >>> from skcomplex.systems.computer_science.evolutionary_computation import EvolutionaryOptimisersSystem
            >>> settings = EvolutionaryOptimisersSystem(algorithm='annealing').get_run_settings()
            >>> [spec.label for spec in settings['observables']]
            ['search/lowest']
        """
        settings = super().get_run_settings()
        if self.algorithm == ANNEALING and self.landscape in MEMBER_ENERGIES:
            member = MEMBER_ENERGIES[self.landscape]
            settings['observables'] = (ObservableSpec('search/lowest', member, group='candidates', reduce='min'),)
        return settings
