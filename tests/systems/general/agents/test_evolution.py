"""Test the `_evolution` module of `skcomplex.systems.general.agents`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.engine.compile import PhaseSpec
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.engine.solvers import AgentView, MesaSolver, SystemView
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System
from skcomplex.systems.general.agents import EvolutionInteraction

NEUTRAL_RUNS = 100


class TypeElement(Element):
    """Members with a strategy and a fitness that no rule of their own changes."""

    def __init__(
        self,
        label: str,
        *,
        strategy: object = 0.0,
        fitness: float = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.strategy = strategy
        self.fitness = fitness


class BitStringView(AgentView):
    """View of a member with a strategy and a fitness."""

    strategy: np.ndarray
    fitness: float


class BitStringElement(Element):
    """Members with a strategy of bits and a fitness."""

    def __init__(
        self,
        label: str,
        *,
        strategy: object = 0.0,
        fitness: float = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.strategy = strategy
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[BitStringView]:
        """Return the view of a member, which types its states."""
        return BitStringView


class ScoringInteraction(Interaction):
    """Scoring that sets the fitness of every member to the count of ones of its strategy."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        """Set the scoring of the strategy in its phase."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self.score, phase='scoring')]

    def score(self, agent: BitStringView, system: SystemView) -> None:
        """Set the fitness to the count of ones."""
        agent.fitness = float(np.sum(agent.strategy))


class AliasEvolutionInteraction(EvolutionInteraction):
    """Evolution that reads one of its probabilities under the parameter `alias`."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        renamed: str,
        alias: float,
        phase: str | None = None,
        replacement: str = 'generational',
        mutation: float = 0.01,
    ) -> None:
        super().__init__(label, target=target, phase=phase, replacement=replacement, mutation=mutation)
        self.renamed = renamed
        self.alias = alias

    def _get_parameter(self, name: str) -> str:
        """Return `alias` for the renamed probability."""
        return 'alias' if name == self.renamed else name


def _run_evolution(evolution: EvolutionInteraction, strategies: np.ndarray, until: int, seed: int) -> Trajectory:
    """Return the trajectory of a group of bit strings that evolve after every scoring."""
    group = BitStringElement('agents', strategy=strategies, size=len(strategies))
    system = System(
        elements=[group],
        interactions=[ScoringInteraction('scoring', target='agents'), evolution],
        solver=MesaSolver(phases=[PhaseSpec('scoring'), PhaseSpec('evolution')]),
    )
    return system.prepare().simulate(until, random_state=seed)


def _strategies(trajectory: Trajectory, frame: int) -> np.ndarray:
    """Return the strategies of the agents at a frame."""
    return np.array(trajectory.get_agents('agents', frame).values['strategy'].tolist(), dtype=float)


@pytest.mark.parametrize('selection', ['tournament', 'roulette', 'rank'])
def test_evolution_interaction_raises_fitness_of_count_of_ones(selection):
    """Test that bit strings evolving on the count of ones for 50 generations raise their mean fitness."""
    evolution = EvolutionInteraction('evolution', target='agents', selection=selection, phase='evolution')
    initial = np.random.default_rng(0).integers(0, 2, size=(40, 20)).astype(float)
    trajectory = _run_evolution(evolution, initial, 50, 0)
    assert _strategies(trajectory, 50).sum(axis=1).mean() > initial.sum(axis=1).mean() + 4


@pytest.mark.parametrize('replacement', ['moran', 'death-birth'])
def test_evolution_interaction_overlapping_replacements_raise_fitness(replacement):
    """Test that bit strings under Moran or death-birth replacement raise their mean fitness."""
    evolution = EvolutionInteraction(
        'evolution',
        target='agents',
        selection='rank',
        replacement=replacement,
        death=0.5,
        phase='evolution',
    )
    initial = np.random.default_rng(0).integers(0, 2, size=(40, 20)).astype(float)
    trajectory = _run_evolution(evolution, initial, 50, 0)
    assert _strategies(trajectory, 50).sum(axis=1).mean() > initial.sum(axis=1).mean() + 4


@pytest.mark.parametrize('replacement', ['moran', 'death-birth'])
def test_evolution_interaction_neutral_fixation_has_the_initial_fraction_as_probability(replacement):
    """Test that a neutral type fixes under Moran or death-birth replacement with the probability of its fraction."""
    evolution = EvolutionInteraction(
        'evolution',
        target='agents',
        selection='roulette',
        replacement=replacement,
        mutation=0.0,
        death=0.3,
    )
    initial = (np.arange(10) < 3).astype(float)[:, None]
    finals = []
    for seed in range(NEUTRAL_RUNS):
        group = TypeElement('agents', strategy=initial, size=10)
        system = System(elements=[group], interactions=[evolution])
        trajectory = system.prepare().simulate(200, random_state=seed)
        finals.append(_strategies(trajectory, 200).ravel())
    assert all(np.ptp(final) == 0.0 for final in finals)
    wins = np.mean([final[0] for final in finals])
    assert wins == pytest.approx(0.3, abs=3 * np.sqrt(0.3 * 0.7 / NEUTRAL_RUNS))


def test_evolution_interaction_mutates_by_a_function():
    """Test that a mutation function that sets one gene to one gives every child exactly one such gene."""

    def flip(strategy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the strategy with one gene drawn at random set to one."""
        strategy[rng.integers(len(strategy))] = 1.0
        return strategy

    evolution = EvolutionInteraction('evolution', target='agents', phase='evolution', mutation=flip, crossover=0.0)
    children = _strategies(_run_evolution(evolution, np.zeros((30, 8)), 1, 0), 1)
    assert children.sum(axis=1).tolist() == [1.0] * 30


def test_evolution_interaction_repeats_with_one_seed():
    """Test that two runs with the same seed give the same strategies."""
    evolution = EvolutionInteraction('evolution', target='agents', phase='evolution', mutation=0.05)
    initial = np.random.default_rng(1).integers(0, 2, size=(10, 8)).astype(float)
    first, second = (_strategies(_run_evolution(evolution, initial, 20, 3), 20) for _ in range(2))
    assert np.array_equal(first, second)


def test_evolution_interaction_mutates_real_strategies_by_gaussian_noise():
    """Test that real strategies with full mutation change by Gaussian noise of the given scale."""
    evolution = EvolutionInteraction(
        'evolution',
        target='agents',
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
    evolution = EvolutionInteraction('evolution', target='agents', phase='evolution', mutation=1.0, elite=1)
    initial = np.zeros((6, 4))
    initial[3] = 1.0
    assert _strategies(_run_evolution(evolution, initial, 1, 0), 1)[0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_evolution_interaction_truncation_redraws_the_least_fit_every_period_and_resets_states():
    """Test that truncation with redraw replaces the least fit only every period and resets their states."""
    evolution = EvolutionInteraction(
        'evolution',
        target='agents',
        replacement='truncation',
        fraction=0.25,
        mutation=1.0,
        change='any',
        alphabet=1000,
        period=2,
        reset={'fitness': -1.0},
    )
    group = TypeElement('agents', strategy=np.arange(8.0)[:, None] * 0 - 1, fitness=np.arange(8.0), size=8)
    system = System(elements=[group], interactions=[evolution])
    trajectory = system.prepare().simulate(2, random_state=0)
    assert np.all(_strategies(trajectory, 1) == -1.0)
    final = trajectory.get_agents('agents', 2).values
    changed = _strategies(trajectory, 2).ravel() != -1.0
    assert changed.tolist() == [True, True] + [False] * 6
    assert final['fitness'].tolist()[:3] == [-1.0, -1.0, 2.0]


def test_evolution_interaction_mutates_to_new_labels_and_by_steps():
    """Test that mutation to new labels exceeds every label held, and steps move a gene by one."""
    for change, check in (('new', lambda values: values.min() > 3.0), ('step', lambda values: set(values) <= {2, 4})):
        evolution = EvolutionInteraction(
            'evolution',
            target='agents',
            fitness=None,
            selection='proportional',
            alphabet=None,
            change=change,
            mutation=1.0,
            crossover=0.0,
        )
        group = TypeElement('agents', strategy=np.full((6, 1), 3.0), size=6)
        system = System(elements=[group], interactions=[evolution])
        trajectory = system.prepare().simulate(1, random_state=0)
        assert check(_strategies(trajectory, 1).ravel())


def test_evolution_interaction_best_mutant_climbs_an_evaluated_landscape():
    """Test that the best-mutant replacement never lowers the evaluated fitness of a common strategy."""
    evolution = EvolutionInteraction(
        'evolution',
        target='agents',
        replacement='best-mutant',
        evaluate=lambda strategies, system: strategies.sum(axis=1),
        mutation=0.02,
    )
    group = TypeElement('agents', strategy=np.zeros((5, 10)), size=5)
    system = System(elements=[group], interactions=[evolution])
    trajectory = system.prepare().simulate(60, random_state=0)
    fitness = [trajectory.get_agents('agents', frame).values['fitness'][0] for frame in range(1, 61)]
    assert np.all(np.diff(fitness) >= 0)
    assert fitness[-1] > 2
    assert np.ptp(_strategies(trajectory, 60), axis=0).max() == 0.0


def test_evolution_interaction_one_gene_changes_and_parents_include_the_dead():
    """Test that the change of one gene alters exactly one gene of every offspring in death-birth among all."""
    evolution = EvolutionInteraction(
        'evolution',
        target='agents',
        selection='proportional',
        replacement='death-birth',
        death=1.0,
        parents='all',
        change='one',
        mutation=1.0,
    )
    group = TypeElement('agents', strategy=np.zeros((10, 7)), size=10)
    system = System(elements=[group], interactions=[evolution])
    trajectory = system.prepare().simulate(1, random_state=0)
    assert _strategies(trajectory, 1).sum(axis=1).tolist() == [1.0] * 10


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'selection': 'lottery'}, 'Unknown selection'),
        ({'replacement': 'steady'}, 'Unknown replacement'),
        ({'mutation': 1.5}, 'mutation probability'),
        ({'mutation': 'flip'}, 'mutation probability'),
        ({'death': 2.0}, 'death probability'),
        ({'crossover': -0.1}, 'crossover probability'),
        ({'elite': -1}, 'elite'),
        ({'tournament': 0}, 'tournament'),
        ({'alphabet': 1}, 'alphabet'),
        ({'change': 'swap'}, 'Unknown change'),
        ({'parents': 'elders'}, 'Unknown parents'),
        ({'fraction': 1.5}, 'fraction probability'),
        ({'period': 0}, 'period'),
        ({'evaluate': 'sum'}, 'evaluation'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that an unknown selection, a probability outside `[0, 1]`, or an invalid count is rejected."""
    with pytest.raises(ValidationError, match=message):
        EvolutionInteraction('evolution', target='agents', **arguments).prepare()


@pytest.mark.parametrize(
    ('replacement', 'name'),
    [('generational', 'crossover'), ('death-birth', 'death'), ('truncation', 'fraction')],
)
def test_value_reads_a_renamed_probability(replacement, name):
    """Test that an evolution that reads a probability under the parameter `alias` evolves as the plain evolution."""
    initial = np.random.default_rng(1).integers(0, 2, size=(10, 8)).astype(float)
    arguments = {'phase': 'evolution', 'replacement': replacement, 'mutation': 0.05}
    plain = EvolutionInteraction('evolution', target='agents', **{name: 0.5}, **arguments)
    renamed = AliasEvolutionInteraction('evolution', target='agents', renamed=name, alias=0.5, **arguments)
    first, second = (_strategies(_run_evolution(item, initial, 10, 3), 10) for item in (plain, renamed))
    np.testing.assert_array_equal(second, first)
