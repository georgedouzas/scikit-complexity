"""Search processes on a landscape: population optimisers and simulated annealing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, cos, pi

from ....core.components.base import BasePart
from ....core.components.elements import Element, GroupElement, LatticeElement
from ....core.components.interactions import Interaction, OptimizationInteraction, SamplingInteraction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from ....core.components.system import BaseSystem, RunSettings, System
from ....core.errors import ValidationError
from ....core.specs import (
    CovarianceAdaptationSpec,
    DifferentialEvolutionSpec,
    EvolutionStrategySpec,
    ObservableSpec,
    ParticleSwarmSpec,
    RoleSpec,
    StateSpec,
)
from ....core.types import Array


class AnnealedCandidateElement(Element):
    """Candidate solution of the plane, whose coordinates are sampled one state at a time.

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
    """Keep the lowest value of an observable, `best <- value if value < best else best`.

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


ALGORITHMS = {
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
    """A population of candidates that searches a two-dimensional landscape, one generation per step.

    A population optimiser moves the candidates through `OptimizationInteraction`, and reports `search.best`. Simulated
    annealing samples every coordinate through `SamplingInteraction` at a temperature that falls geometrically to 1e-3
    of its start over `generations` steps, and `tracker/record.best` holds the lowest energy it has reached, from the
    observable `search/lowest` that the catalog lists, so a run passes it in `observables` (Raman 2021 §6.3.3.3,
    §6.3.4). The system has the group `candidates`.

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
    ) -> System:
        """Return candidates that anneal one state at a time, and the element keeping the lowest energy they reach."""
        candidates = LatticeElement(
            'candidates',
            element=AnnealedCandidateElement,
            space=LatticeSpace((len(start),)),
            values={'x': start[:, 0], 'y': start[:, 1]},
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
                neighborhood=VonNeumannNeighborhood(n_dim=1),
                cooling=lambda t: COOLED ** (t / generations),
            )
            for state in ('x', 'y')
        ]
        return System(
            label,
            systems=[
                System('search', elements=[candidates], interactions=samplers, time=DiscreteTime()),
                System('tracker', elements=[RunningBestElement('record')], time=DiscreteTime()),
            ],
            interactions=[RunningBestInteraction('update', target='tracker/record', source='search/lowest')],
            time=DiscreteTime(),
        )

    def _build_system(self) -> System:
        """Return a population of candidates that searches a two-dimensional landscape, one generation per step."""
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
        return System(self.label, elements=[candidates], interactions=[search], time=DiscreteTime())

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings, with the lowest candidate energy of simulated annealing.

        A population optimiser keeps the default settings. Simulated annealing nests the search and the tracker of the
        best energy, which run side by side, and records the lowest energy its candidates reach, as `search/lowest`,
        which the tracker reads.

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
