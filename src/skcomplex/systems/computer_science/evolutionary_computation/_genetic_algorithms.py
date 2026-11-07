"""Evolutionary computation systems: Mitchell's genetic algorithm and Gooding's evolution by circumstance of Robby."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components.elements import GroupElement
from ....core.components.interactions import EvolutionInteraction, Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.system import System
from ....core.engine.solvers import SystemView
from ....core.errors import ParameterError
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from ....core.types import IntArray
from ..._catalog import register_system
from ._room_cleaning import RoomCleaningInteraction
from ._strategy import StrategyElement
from ._strategy_view import StrategyView

N_SITUATIONS = 243


N_ACTIONS = 7


REGIMES = ('generational', 'circumstance')


class AgedStrategyView(StrategyView):
    """View of a strategy of Robby with an age, with its states as attributes."""

    age: float


class AgedStrategyElement(StrategyElement):
    """Strategy of Robby with an age, which dies of old age or starves in the evolution by circumstance.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        actions:
            The initial action per situation, an array of 243 integers from 0 to 6.

        score:
            The initial score.

        age:
            The initial age in generations.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import AgedStrategyElement
        >>> AgedStrategyElement('strategy', age=3).age
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'score'}),
        *,
        actions: Annotated[object, StateSpec()] = 0.0,
        score: Annotated[float, StateSpec()] = 0.0,
        age: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual, actions=actions, score=score)
        self.age = age

    @classmethod
    def _get_view(cls) -> type[AgedStrategyView]:
        """Return the view of the aged strategies, which types their states."""
        return AgedStrategyView


class CircumstanceInteraction(Interaction):
    """Evolution by circumstance: strategies die of old age or starve, and random survivors breed in their places.

    Every generation the strategies age by one, the ones that reach `lifespan` die of old age, and the fraction
    `starvation` of the population with the lowest scores starves. Random pairs of survivors, whatever their score,
    exchange the tails of their strategies after a random cut, mutate every action to another one with probability
    `mutation`, and place their child, of age 0, in the place of a dead strategy, until the population is whole again.
    With a `switch` period, the starvation is on for `switch` generations and off for the next `switch`, in turn.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of aged strategies.

        lifespan:
            The age at which a strategy dies.

        starvation:
            The fraction of the population, of the lowest scores, that starves every generation.

        mutation:
            The probability that an action of a child is replaced by another one.

        switch:
            The number of generations after which the starvation switches on or off, or `None` to keep it on.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import CircumstanceInteraction
        >>> CircumstanceInteraction('circumstance', target='strategies').lifespan
        7
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        lifespan: int = 7,
        starvation: float = 0.3,
        mutation: float = 0.005,
        switch: int | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.lifespan = lifespan
        self.starvation = starvation
        self.mutation = mutation
        self.switch = switch

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the deaths and births of a generation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule of the generation.
        """
        return [RuleLaw(self._generation)]

    def _generation(self, system: SystemView) -> None:
        """Age the strategies, remove the old and starving ones, and fill their places with children of survivors."""
        agents = system.agents('target', AgedStrategyView)
        rng = system.rng
        ages = np.array([agent.age for agent in agents]) + 1
        scores = np.array([agent.score for agent in agents])
        dead = ages >= self.lifespan
        if self._starving(system.time):
            dead[np.argsort(scores, kind='stable')[: round(self.starvation * len(agents))]] = True
        survivors = np.flatnonzero(~dead)
        if not survivors.size:
            survivors = np.arange(len(agents))
        strategies = np.array([agent.actions for agent in agents], dtype=np.int64)
        for agent, age in zip(agents, ages, strict=True):
            agent.age = float(age)
        for index in np.flatnonzero(dead):
            first, second = rng.choice(survivors, size=2)
            agents[index].actions = self._child(strategies[first], strategies[second], rng)
            agents[index].age = 0.0

    def _starving(self, time: float) -> bool:
        """Return whether the starvation is on in the generation of a time."""
        return self.switch is None or int(time // self.switch) % 2 == 0

    def _child(self, first: IntArray, second: IntArray, rng: np.random.Generator) -> IntArray:
        """Return the child of two strategies, the head of one and the tail of the other, with mutated actions."""
        cut = int(rng.integers(1, first.size))
        child = np.concatenate([first[:cut], second[cut:]])
        mutated = rng.random(child.size) < self.mutation
        shifts = rng.integers(1, N_ACTIONS, child.size)
        return np.where(mutated, (child + shifts) % N_ACTIONS, child)


@register_system(
    field='computer_science/evolutionary_computation',
    tags=('genetic algorithm', 'agents', 'evolution', 'robot'),
    references=('Mitchell 2009', 'Gooding 2019'),
    defaults={'until': 20},
)
def robby_ga(
    n_strategies: int = 50,
    *,
    mutation: float = 0.005,
    sessions: int = 10,
    n_actions: int = 100,
    regime: str = 'generational',
    lifespan: int = 7,
    starvation: float = 0.3,
    switch: int | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Mitchell's genetic algorithm that evolves strategies for Robby, a robot that collects cans in a room.

    Each step is one generation. Parents are selected with probabilities proportional to their rank, every pair of
    parents exchanges the tails of their strategies after a random cut, and every action of a child mutates to another
    action with probability `mutation`. Then every child is scored in random rooms. Random strategies score far below
    zero because they bump into walls, and evolution finds strategies that avoid walls and collect cans.

    Gooding's evolution by circumstance, the regime `'circumstance'`, replaces the selection with deaths: every
    strategy starts at a random age, dies at the age `lifespan`, and the fraction `starvation` with the lowest scores
    starves every generation. Random pairs of survivors, whatever their scores, breed children in the places of the
    dead. The starvation is the only selection, so with it the scores rise as under the genetic algorithm, and without
    it they drift. With a `switch` period the starvation switches on and off every `switch` generations, as in
    Gooding's runs that switch it every 3000 iterations.

    Args:
        n_strategies:
            The size of the population.

        mutation:
            The probability that an action of a child is replaced by another one.

        sessions:
            The number of rooms a strategy cleans to be scored.

        n_actions:
            The number of actions in a session.

        regime:
            The evolution, `'generational'` for Mitchell's genetic algorithm or `'circumstance'` for Gooding's
            evolution by circumstance.

        lifespan:
            The age at which a strategy dies, in the evolution by circumstance.

        starvation:
            The fraction of the lowest scores that starves every generation, in the evolution by circumstance.

        switch:
            The number of generations after which the starvation switches on or off, or `None` to keep it on, in the
            evolution by circumstance.

        random_state:
            The seed or generator of the random first generation and of its scores.

    Returns:
        system:
            The system, with the group `strategies` of states `actions` and `score`, and `age` in the evolution by
            circumstance.

    Raises:
        ParameterError:
            The regime is unknown.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import robby_ga
        >>> robby_ga(5).elements[0].values['actions'].shape
        (5, 243)
    """
    if regime not in REGIMES:
        error_msg = f'Unknown regime {regime!r}. Use one of {", ".join(REGIMES)}.'
        raise ParameterError(error_msg)
    rng = np.random.default_rng(random_state)
    evolution = EvolutionInteraction(
        'evolution',
        target='strategies',
        strategy='actions',
        fitness='score',
        selection='rank',
        crossover=1.0,
        mutation=mutation,
        alphabet=N_ACTIONS,
    )
    scoring = RoomCleaningInteraction('scoring', target='strategies', sessions=sessions, n_actions=n_actions)
    actions = rng.integers(0, N_ACTIONS, size=(n_strategies, N_SITUATIONS))
    scores = scoring.score(actions, rng)
    if regime == 'circumstance':
        circumstance = CircumstanceInteraction(
            'circumstance',
            target='strategies',
            lifespan=lifespan,
            starvation=starvation,
            mutation=mutation,
            switch=switch,
        )
        ages = rng.integers(0, lifespan, size=n_strategies).astype(float)
        strategies = GroupElement(
            'strategies',
            element=AgedStrategyElement,
            size=n_strategies,
            values={'actions': actions, 'score': scores, 'age': ages},
        )
        return System(elements=[strategies], interactions=[circumstance, scoring])
    return System(
        elements=[
            GroupElement(
                'strategies',
                element=StrategyElement,
                size=n_strategies,
                values={'actions': actions, 'score': scores},
            ),
        ],
        interactions=[evolution, scoring],
    )
