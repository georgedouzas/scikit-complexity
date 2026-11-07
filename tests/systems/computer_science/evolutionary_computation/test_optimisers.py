"""Test the `_optimisers` module of `skcomplex.systems.computer_science.evolutionary_computation`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.computer_science.evolutionary_computation import EvolutionaryOptimisersSystem

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
