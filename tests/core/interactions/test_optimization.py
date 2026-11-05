"""Test the `_optimization` module of `skcomplex.core.interactions`."""

from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import OptimizationInteraction
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import (
    CovarianceAdaptationSpec,
    DifferentialEvolutionSpec,
    EvolutionStrategySpec,
    ParticleSwarmSpec,
    StateSpec,
)


class CandidateElement(Element):
    """Candidate with a vector state."""

    def __init__(self, label: str, x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = (0.0, 0.0)) -> None:
        super().__init__(label)
        self.x = x


def _rastrigin(values: np.ndarray) -> np.ndarray:
    """Return the two-dimensional Rastrigin function of every candidate."""
    return 20.0 + (values**2 - 10.0 * np.cos(2.0 * np.pi * values)).sum(axis=1)


def _system(algorithm: object, count: int = 40) -> System:
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
    return System(elements=[group], interactions=[search], time=DiscreteTime())


def test_run_differential_evolution_on_rastrigin() -> None:
    """Test that the best value is below 1e-3 by step 50 and never increases."""
    trajectory = Simulation(_system(DifferentialEvolutionSpec()), random_state=1).prepare().run(until=50).trajectory_
    best = trajectory.values('search.best')[1:]
    assert best[-1] < 1e-3
    assert np.all(np.diff(best) <= 0)
    assert np.asarray(trajectory.agents('swarm', 50).values['x'].tolist()).shape == (40, 2)


@pytest.mark.parametrize(
    'algorithm',
    [DifferentialEvolutionSpec(), ParticleSwarmSpec(), EvolutionStrategySpec(), CovarianceAdaptationSpec()],
)
def test_run_same_seed_identical(algorithm: object) -> None:
    """Test that runs with the same seed are identical and the best value never increases."""
    first, second = (
        Simulation(_system(algorithm), random_state=1).prepare().run(until=10).trajectory_ for _ in range(2)
    )
    assert np.array_equal(first.values('search.best')[1:], second.values('search.best')[1:])
    first_final = np.asarray(first.agents('swarm', 10).values['x'].tolist())
    second_final = np.asarray(second.agents('swarm', 10).values['x'].tolist())
    assert np.array_equal(first_final, second_final)
    assert np.all(np.diff(first.values('search.best')[1:]) <= 0)


def test_run_algorithm_that_converges_keeps_its_population() -> None:
    """Test that the steps after the algorithm stops itself report the same values and keep the candidates."""
    trajectory = Simulation(_system(CovarianceAdaptationSpec()), random_state=1).prepare().run(until=400).trajectory_
    best = trajectory.values('search.best')
    assert best[-1] == best[-2]
    assert np.array_equal(
        trajectory.agents('swarm', 399).values['x'].tolist(),
        trajectory.agents('swarm', 400).values['x'].tolist(),
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
