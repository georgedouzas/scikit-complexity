"""Test the `_optimisers` module of `skcomplex.systems.computer_science.evolutionary_computation`."""

from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec
from skcomplex.systems.computer_science.evolutionary_computation import (
    CovarianceAdaptationSpec,
    DifferentialEvolutionSpec,
    EvolutionaryOptimisersSystem,
    EvolutionStrategySpec,
    OptimizationInteraction,
    ParticleSwarmSpec,
)

PATHS = {
    'differential-evolution': 'search.best',
    'particle-swarm': 'search.best',
    'annealing': 'tracker/record.best',
}


def _best(algorithm: str, landscape: str = 'rastrigin') -> np.ndarray:
    """Return the best value of a run of 100 generations after the initial frame."""
    system = EvolutionaryOptimisersSystem(algorithm=algorithm, landscape=landscape, random_state=1)
    return simulate(system, until=100, random_state=1).values(PATHS.get(algorithm, 'search.best'))[1:]


@pytest.mark.parametrize('algorithm', PATHS)
def test_evolutionary_optimisers_best_never_increases(algorithm: str) -> None:
    """Test that the best value of every algorithm never increases with the same seed."""
    best = _best(algorithm)
    assert np.all(np.diff(best) <= 0)


def test_evolutionary_optimisers_differential_evolution_reaches_the_minimum() -> None:
    """Test that differential evolution ends below 1e-3 on Rastrigin."""
    assert _best('differential-evolution')[-1] < 1e-3


@pytest.mark.parametrize('algorithm', ['evolution-strategy', 'covariance-adaptation', 'annealing'])
def test_evolutionary_optimisers_rosenbrock(algorithm: str) -> None:
    """Test that the other algorithms search the Rosenbrock landscape, whose start they improve on."""
    best = _best(algorithm, 'rosenbrock')
    assert best[-1] < best[0] or best[-1] < 1e-3


def test_evolutionary_optimisers_unknown_names() -> None:
    """Test that an unknown algorithm or landscape is refused."""
    with pytest.raises(ValidationError, match='algorithm'):
        EvolutionaryOptimisersSystem(algorithm='ga').prepare()
    with pytest.raises(ValidationError, match='landscape'):
        EvolutionaryOptimisersSystem(landscape='sphere').prepare()


class CandidateElement(Element):
    """Candidate with a vector state."""

    def __init__(self, label: str, x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = (0.0, 0.0)) -> None:
        super().__init__(label)
        self.x = x


def _rastrigin(values: np.ndarray) -> np.ndarray:
    """Return the two-dimensional Rastrigin function of every candidate."""
    return 20.0 + (values**2 - 10.0 * np.cos(2.0 * np.pi * values)).sum(axis=1)


def _system(algorithm: object, count: int = 40) -> CompositeSystem:
    """Return a population on the Rastrigin function with an algorithm."""
    start = np.random.default_rng(0).uniform(-5.12, 5.12, (count, 2))
    group = GroupElement('swarm', element=CandidateElement, size=count, values={'x': start})
    search = OptimizationInteraction(
        'search',
        target='swarm',
        state='x',
        objective=_rastrigin,
        bounds=(-5.12, 5.12),
        algorithm=algorithm,
    )
    return CompositeSystem(elements=[group], interactions=[search], time=DiscreteTime())


def test_run_differential_evolution_on_rastrigin() -> None:
    """Test that the best value is below 1e-3 by step 50 and never increases."""
    trajectory = simulate(_system(DifferentialEvolutionSpec()), until=50, random_state=1)
    best = trajectory.values('search.best')[1:]
    assert best[-1] < 1e-3
    assert np.all(np.diff(best) <= 0)
    assert np.asarray(trajectory.get_agents('swarm', 50).values['x'].tolist()).shape == (40, 2)


@pytest.mark.parametrize(
    'algorithm',
    [DifferentialEvolutionSpec(), ParticleSwarmSpec(), EvolutionStrategySpec(), CovarianceAdaptationSpec()],
)
def test_run_same_seed_identical(algorithm: object) -> None:
    """Test that runs with the same seed are identical and the best value never increases."""
    first, second = (simulate(_system(algorithm), until=10, random_state=1) for _ in range(2))
    assert np.array_equal(first.values('search.best')[1:], second.values('search.best')[1:])
    first_final = np.asarray(first.get_agents('swarm', 10).values['x'].tolist())
    second_final = np.asarray(second.get_agents('swarm', 10).values['x'].tolist())
    assert np.array_equal(first_final, second_final)
    assert np.all(np.diff(first.values('search.best')[1:]) <= 0)


def test_run_algorithm_that_converges_keeps_its_population() -> None:
    """Test that the steps after the algorithm stops itself report the same values and keep the candidates."""
    trajectory = simulate(_system(CovarianceAdaptationSpec()), until=400, random_state=1)
    best = trajectory.values('search.best')
    assert best[-1] == best[-2]
    assert np.array_equal(
        trajectory.get_agents('swarm', 399).values['x'].tolist(),
        trajectory.get_agents('swarm', 400).values['x'].tolist(),
    )


def test_check_algorithm_not_a_spec() -> None:
    """Test that an algorithm that is not a spec raises an error that lists the specs."""
    interaction = OptimizationInteraction(
        'search',
        target='swarm',
        state='x',
        objective=_rastrigin,
        bounds=(-1, 1),
        algorithm='ga',
    )
    with pytest.raises(ValidationError, match=r'DifferentialEvolutionSpec.*CovarianceAdaptationSpec'):
        interaction.prepare()
