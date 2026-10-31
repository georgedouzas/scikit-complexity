"""Test the `_evolution` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import EvolutionInteraction
from skcomplex.core.laws import BaseLaw, RuleLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.engine.solvers import AgentView, ModelView
from skcomplex.errors import ValidationError
from skcomplex.specs import PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec


class BitStringView(AgentView):
    """View of a member with a strategy and a fitness."""

    strategy: np.ndarray
    fitness: float


class BitStringElement(Element):
    """Member whose fitness is the count of ones of its strategy."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[object, StateSpec()] = 0.0,
        fitness: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[BitStringView]:
        """Return the view of a member, which types its states."""
        return BitStringView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the scoring of the strategy."""
        return [RuleLaw(self.score, phase='scoring')]

    def score(self, agent: BitStringView, model: ModelView) -> None:
        """Set the fitness to the count of ones."""
        agent.fitness = float(np.sum(agent.strategy))


def _run_evolution(evolution: EvolutionInteraction, strategies: np.ndarray, until: int, seed: int) -> Trajectory:
    """Return the trajectory of a group of bit strings that evolve after every scoring."""
    group = GroupElement('agents', element=BitStringElement, size=len(strategies), values={'strategy': strategies})
    schedule = PhasesScheduleSpec([PhaseSpec('scoring'), PhaseSpec('evolution')])
    system = System(elements=[group], interactions=[evolution], time=DiscreteTime(schedule=schedule))
    return Simulation(system, random_state=seed).check().prepare().run(until=until).trajectory_


def _strategies(trajectory: Trajectory, frame: int) -> np.ndarray:
    """Return the strategies of the agents at a frame."""
    return np.array(trajectory.agents('agents', frame).values['strategy'].tolist(), dtype=float)


@pytest.mark.parametrize('selection', ['tournament', 'roulette'])
def test_evolution_interaction_raises_fitness_of_count_of_ones(selection):
    """Test that bit strings evolving on the count of ones for 50 generations raise their mean fitness."""
    evolution = EvolutionInteraction('evolution', members='agents', selection=selection, phase='evolution')
    initial = np.random.default_rng(0).integers(0, 2, size=(40, 20)).astype(float)
    trajectory = _run_evolution(evolution, initial, 50, 0)
    assert _strategies(trajectory, 50).sum(axis=1).mean() > initial.sum(axis=1).mean() + 4


def test_evolution_interaction_repeats_with_one_seed():
    """Test that two runs with the same seed give the same strategies."""
    evolution = EvolutionInteraction('evolution', members='agents', phase='evolution', mutation=0.05)
    initial = np.random.default_rng(1).integers(0, 2, size=(10, 8)).astype(float)
    first, second = (_strategies(_run_evolution(evolution, initial, 20, 3), 20) for _ in range(2))
    assert np.array_equal(first, second)


def test_evolution_interaction_mutates_real_strategies_by_gaussian_noise():
    """Test that real strategies with full mutation change by Gaussian noise of the given scale."""
    evolution = EvolutionInteraction(
        'evolution',
        members='agents',
        phase='evolution',
        alphabet=None,
        mutation=1.0,
        crossover=0.0,
        scale=0.5,
    )
    trajectory = _run_evolution(evolution, np.zeros((200, 10)), 1, 0)
    assert abs(_strategies(trajectory, 1).std() - 0.5) < 0.05


def test_evolution_interaction_keeps_the_elite():
    """Test that the fittest strategy passes unchanged into the next generation."""
    evolution = EvolutionInteraction('evolution', members='agents', phase='evolution', mutation=1.0, elite=1)
    initial = np.zeros((6, 4))
    initial[3] = 1.0
    assert _strategies(_run_evolution(evolution, initial, 1, 0), 1)[0].tolist() == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'selection': 'lottery'}, 'Unknown selection'),
        ({'mutation': 1.5}, 'mutation probability'),
        ({'crossover': -0.1}, 'crossover probability'),
        ({'elite': -1}, 'elite'),
        ({'tournament': 0}, 'tournament'),
        ({'alphabet': 1}, 'alphabet'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that an unknown selection, a probability outside `[0, 1]`, or an invalid count is rejected."""
    with pytest.raises(ValidationError, match=message):
        EvolutionInteraction('evolution', members='agents', **arguments).check()
