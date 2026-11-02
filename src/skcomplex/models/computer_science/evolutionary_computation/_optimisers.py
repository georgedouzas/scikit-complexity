"""Search processes on a landscape: population optimisers and simulated annealing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, pi

from ....core.elements import GroupElement, LatticeElement
from ....core.interactions import OptimizationInteraction, SamplingInteraction
from ....core.spaces import DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from ....core.system import System
from ....errors import ValidationError
from ....specs import (
    CovarianceAdaptationSpec,
    DifferentialEvolutionSpec,
    EvolutionStrategySpec,
    ObservableSpec,
    ParticleSwarmSpec,
)
from ....types import Array
from ..._catalog import register_model
from ._annealed_candidate import AnnealedCandidateElement
from ._running_best import RunningBestElement
from ._running_best_update import RunningBestInteraction
from ._search_candidate import SearchCandidateElement

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
MEMBER_ENERGIES = {_rastrigin_energy: _rastrigin_member, _rosenbrock_energy: _rosenbrock_member}


def _search_observables(system: System) -> list[ObservableSpec]:
    """Return the lowest energy of the candidates of simulated annealing, which a population optimiser does not need."""
    if not system.systems:
        return []
    energy = MEMBER_ENERGIES[system.systems[0].interactions[0].energy]
    return [ObservableSpec('search/lowest', energy, group='candidates', reduce='min')]


def _annealing(
    start: Array,
    energy: Callable[[SimpleNamespace], Expression],
    temperature: float,
    width: float,
    generations: float,
) -> System:
    """Return candidates that anneal one state at a time, and the element that keeps the lowest energy they reach."""
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
            energy=energy,
            move='gaussian',
            width=width,
            temperature=temperature,
            schedule='colored',
            neighborhood=VonNeumannNeighborhood(n_dim=1),
            cooling=lambda t: COOLED ** (t / generations),
        )
        for state in ('x', 'y')
    ]
    return System(
        systems=[
            System('search', elements=[candidates], interactions=samplers, time=DiscreteTime()),
            System('tracker', elements=[RunningBestElement('record')], time=DiscreteTime()),
        ],
        interactions=[RunningBestInteraction('update', target='tracker/record', source='search/lowest')],
        time=DiscreteTime(),
    )


@register_model(
    field='computer_science/evolutionary_computation',
    tags=('optimisation', 'evolutionary algorithm', 'annealing', 'fitness landscape'),
    references=('Raman 2021', 'Storn and Price 1997', 'Kirkpatrick, Gelatt, and Vecchi 1983'),
    defaults={'until': 100},
    observables=_search_observables,
)
def evolutionary_optimisers(
    n_candidates: int = 40,
    *,
    algorithm: str = 'differential-evolution',
    landscape: str = 'rastrigin',
    temperature: float = 5.0,
    width: float = 0.5,
    generations: float = 100.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a population of candidates that searches a two-dimensional landscape, one generation per step.

    A population optimiser moves the candidates through `OptimizationInteraction`, and reports `search.best`. Simulated
    annealing samples every coordinate through `SamplingInteraction` at a temperature that falls geometrically to 1e-3
    of its start over `generations` steps, and `tracker/record.best` holds the lowest energy it has reached, from the
    observable `search/lowest` that the catalog lists, so a run passes it in `observables` (Raman 2021 §6.3.3.3,
    §6.3.4).

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

    Returns:
        system:
            The model, with the group `candidates`.

    Raises:
        ValidationError:
            The algorithm or the landscape is unknown.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import evolutionary_optimisers
        >>> evolutionary_optimisers(10).interactions[0].label
        'search'
    """
    if (algorithm not in ALGORITHMS and algorithm != ANNEALING) or landscape not in LANDSCAPES:
        names = ', '.join((*ALGORITHMS, ANNEALING))
        error_msg = (
            f'evolutionary_optimisers has algorithm={algorithm!r} and landscape={landscape!r}. Use one of {names}, '
            f'and one of {", ".join(LANDSCAPES)}.'
        )
        raise ValidationError(error_msg)
    energy, objective, bound = LANDSCAPES[landscape]
    start = np.random.default_rng(random_state).uniform(-bound, bound, (n_candidates, N_DIM))
    if algorithm == ANNEALING:
        return _annealing(start, energy, temperature, width, generations)
    candidates = GroupElement('candidates', element=SearchCandidateElement, size=n_candidates, values={'x': start})
    search = OptimizationInteraction(
        'search',
        target='candidates',
        state='x',
        objective=objective,
        bounds=(-bound, bound),
        algorithm=ALGORITHMS[algorithm],
    )
    return System(elements=[candidates], interactions=[search], time=DiscreteTime())
