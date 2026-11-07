"""Econophysics systems: the Kolkata Paise Restaurant problem of choosing among restaurants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import ChoiceInteraction, GroupingInteraction
from ....core.components.laws import BaseLaw, MatchLaw, RuleLaw, UpdateLaw
from ....core.components.system import BaseSystem, System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import StateSpec, VisualSpec


class DinerView(AgentView):
    """Diner as the rules see it, with its restaurant, the crowd there, and whether it was served.

    It reads its `restaurant`, the `crowd` there, and whether it was `served`.
    """

    restaurant: float
    crowd: float
    served: float


class DinerElement(Element):
    """Diner of the Kolkata Paise Restaurant problem, who goes to one restaurant every evening.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        restaurant:
            The restaurant chosen last, from 0 to the number of restaurants minus one.

        crowd:
            The number of diners at that restaurant, 0 before the first evening.

        served:
            Whether the diner got a meal, 1 or 0.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.econophysics import DinerElement
        >>> GroupElement('diners', element=DinerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        restaurant: Annotated[float, StateSpec(domain=NN)] = 0,
        crowd: Annotated[float, StateSpec(domain=NN)] = 0,
        served: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.restaurant = restaurant
        self.crowd = crowd
        self.served = served

    @classmethod
    def _get_view(cls) -> type[DinerView]:
        """Return the view of the agents of the element."""
        return DinerView


class CrowdAvoidanceInteraction(ChoiceInteraction):
    """Stochastic crowd avoidance of diners, who leave crowded restaurants.

    A diner returns to last evening's restaurant with probability `1 / n`, with `n` the crowd it found there, and
    otherwise moves to one of the other restaurants at random, which raises the utilization to about 0.8. It is a
    Metropolis choice of intensity 1 where every other restaurant has utility 0 and staying has the utility
    `-log(1 - 1 / n)`, so a proposed move is accepted with probability `1 - 1 / n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of diners.

        options:
            The restaurants, from 0 to their number minus one.

        state:
            The state of the diners that holds their restaurant.

        intensity:
            The intensity of choice, 1 for the stay probability `1 / n`.

        activity:
            The probability that a diner revises its choice in an evening.

        phase:
            The phase the choice runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import CrowdAvoidanceInteraction
        >>> CrowdAvoidanceInteraction('choice', target='diners', options=(0.0, 1.0)).rule
        'metropolis'
    """

    def __init__(
        self,
        label: str,
        *,
        target: str,
        options: Sequence[float],
        state: str = 'restaurant',
        intensity: float = 1.0,
        activity: float = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            options=options,
            utility=self._get_utility,
            rule='metropolis',
            intensity=intensity,
            activity=activity,
            form='agent',
            phase=phase,
        )

    def _get_utility(self, agent: DinerView, system: SystemViewProtocol, option: float) -> float:
        """Return 0 for another restaurant and `-log(1 - 1 / n)` for staying at a crowd of `n`."""
        if option != getattr(agent, self.state) or agent.crowd < 1:
            return 0.0
        if agent.crowd == 1:
            return math.inf
        return -math.log1p(-1 / agent.crowd)


class RestaurantsInteraction(GroupingInteraction):
    """Restaurants that each serve one meal every evening, to one of the diners who come chosen at random.

    The diners are grouped by their restaurant. Every diner learns the crowd at its restaurant, and the first of the
    group in a random order is served. The interaction reports the fraction of restaurants that serve a meal.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of diners.

        n_restaurants:
            The number of restaurants.

        by:
            The state of the diners that holds their restaurant.

        phase:
            The phase the restaurants serve in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import RestaurantsInteraction
        >>> RestaurantsInteraction('restaurants', target='diners', n_restaurants=5).groups
        'state'
    """

    law_classes = (MatchLaw, RuleLaw, UpdateLaw)

    reports: ClassVar[tuple[str, ...]] = ('utilization',)

    def __init__(
        self,
        label: str,
        *,
        target: str,
        n_restaurants: int = 100,
        by: str = 'restaurant',
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='state', by=by, phase=phase)
        self.n_restaurants = n_restaurants

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Tell every diner of a restaurant its crowd and serve the first diner.

        Args:
            members:
                The diners at one restaurant, in random order.

            system:
                The system view of the restaurants.
        """
        diners = [member for member in members if isinstance(member, DinerView)]
        for position, member in enumerate(diners):
            member.crowd = float(len(diners))
            member.served = float(position == 0)

    def report(self, system: SystemViewProtocol) -> None:
        """Report the fraction of restaurants with at least one diner.

        Args:
            system:
                The system view of the restaurants.
        """
        visited = {float(getattr(agent, str(self.by))) for agent in system.agents('target')}
        system.report('utilization', len(visited) / self.n_restaurants)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that serves the diners and the rule that reports the utilization.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The serving and the reporting rules.
        """
        return [*super()._get_laws(namespace), RuleLaw(self.report, phase=self.phase)]


STRATEGIES = ('random', 'crowd_avoiding')


class KolkataRestaurantSystem(BaseSystem):
    """The Kolkata Paise Restaurant problem, where diners spread over restaurants that serve one meal each.

    With random choice the expected utilization of `N` restaurants by `N` diners is `1 - 1/e`, about 0.632, and with the
    stochastic crowd avoiding strategy it settles near 0.8. The system has the group `diners`, the choice `choice` of a
    restaurant every evening, uniform or by crowd avoidance, and the interaction `restaurants`, which reports the
    fraction of restaurants that serve a meal `restaurants.utilization`.

    Args:
        n_agents:
            The number of diners.

        n_restaurants:
            The number of restaurants, or `None` for as many as diners.

        strategy:
            `'random'` or `'crowd_avoiding'`.

        random_state:
            The seed or generator of the first choices.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValueError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import KolkataRestaurantSystem
        >>> type(KolkataRestaurantSystem(10, strategy='crowd_avoiding').prepare().system_.interactions[0]).__name__
        'CrowdAvoidanceInteraction'
    """

    field = 'economics/econophysics'
    topics = ('kolkata paise restaurant', 'agents', 'resource allocation', 'minority game')
    references = ('Chakrabarti et al. 2009', 'Ghosh et al. 2010', 'Carmichael and Hadzikadic 2019 ch. 9')
    until = 100

    def __init__(
        self,
        n_agents: int = 100,
        *,
        n_restaurants: int | None = None,
        strategy: str = 'random',
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.n_restaurants = n_restaurants
        self.strategy = strategy
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        n_restaurants = self.n_restaurants
        if self.strategy not in STRATEGIES:
            error_msg = f'Unknown strategy {self.strategy!r}. Use one of {STRATEGIES}.'
            raise ValueError(error_msg)
        n_restaurants = self.n_agents if n_restaurants is None else n_restaurants
        options = [float(restaurant) for restaurant in range(n_restaurants)]
        choices = np.random.default_rng(self.random_state).integers(n_restaurants, size=self.n_agents).astype(float)
        diners = GroupElement('diners', element=DinerElement, size=self.n_agents, values={'restaurant': choices})
        choice = (
            CrowdAvoidanceInteraction('choice', target='diners', options=options)
            if self.strategy == 'crowd_avoiding'
            else ChoiceInteraction(
                'choice',
                target='diners',
                state='restaurant',
                options=options,
                utility=lambda agent, system, option: 0.0,
                intensity=0.0,
                form='agent',
            )
        )
        restaurants = RestaurantsInteraction('restaurants', target='diners', n_restaurants=n_restaurants)
        return System(self.label, elements=[diners], interactions=[choice, restaurants])
