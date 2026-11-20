"""Econophysics systems: the Kolkata Paise Restaurant problem.

Diners choose among restaurants.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.agents import ChoiceInteraction

STRATEGIES = ('random', 'crowd_avoiding')


class DinerView(AgentView):
    """Diner as the rules see it.

    It reads its `restaurant`, the `crowd` there, and whether it was `served`.
    """

    restaurant: float
    crowd: float
    served: float


class DinerElement(Element):
    """Diner of the Kolkata Paise Restaurant problem.

    It goes to one restaurant every evening.

    Args:
        label:
            The label of the agent or group.

        restaurant:
            The restaurant chosen last, from 0 to the number of restaurants minus one.

        crowd:
            The number of diners at that restaurant, 0 before the first evening.

        served:
            Whether the diner got a meal, 1 or 0.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import DinerElement
        >>> DinerElement('diners', size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        *,
        restaurant: Annotated[object, NN] = 0.0,
        crowd: Annotated[object, NN] = 0.0,
        served: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
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
            schedule='synchronous',
            phase=phase,
        )

    def _get_utility(self, agent: DinerView, system: SystemViewProtocol, option: float) -> float:
        """Return the utility of a restaurant.

        It is 0 for another restaurant and `-log(1 - 1 / n)` for staying at a crowd of `n`.
        """
        if option != getattr(agent, self.state) or agent.crowd < 1:
            return 0.0
        if agent.crowd == 1:
            return math.inf
        return -math.log1p(-1 / agent.crowd)


class RestaurantsInteraction(Interaction):
    """Restaurants that each serve one meal every evening.

    The meal goes to one of the diners who come, chosen at random. The diners are grouped by their restaurant. Every
    diner learns the crowd at its restaurant, and the first of the group in a random order is served. The interaction
    reports the fraction of restaurants that serve a meal.

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
        >>> RestaurantsInteraction('restaurants', target='diners', n_restaurants=5).by
        'restaurant'
    """

    law_classes = (UpdateLaw,)

    reports: ClassVar[tuple[str, ...]] = ('utilization',)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        n_restaurants: int = 100,
        by: str = 'restaurant',
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.n_restaurants = n_restaurants
        self.by = by
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the serving and then the reporting of every evening as rules of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [
                UpdateLaw(context, self.serve, phase=self.phase),
                UpdateLaw(context, self.report, phase=self.phase),
            ]

    def serve(self, system: SystemViewProtocol) -> None:
        """Group the diners by their restaurant and serve every restaurant, in random order.

        The diners of a restaurant come in a random order, and the restaurants are visited in a random order.

        Args:
            system:
                The system view of the restaurants.
        """
        shared: dict[float, list[AgentViewProtocol]] = {}
        for agent in system.get_agents('target'):
            if agent.alive:
                shared.setdefault(float(getattr(agent, self.by)), []).append(agent)
        groups = []
        for value in sorted(shared):
            members = shared[value]
            groups.append([members[int(index)] for index in system.rng.permutation(len(members))])
        for index in system.rng.permutation(len(groups)):
            self.act_on(groups[int(index)], system)

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
        visited = {float(getattr(agent, self.by)) for agent in system.get_agents('target')}
        system.report('utilization', len(visited) / self.n_restaurants)


class KolkataRestaurantSystem(BaseSystem):
    """The Kolkata Paise Restaurant problem.

    Diners spread over restaurants that serve one meal each. With random choice the expected utilization of `N`
    restaurants by `N` diners is `1 - 1/e`, about 0.632, and with the stochastic crowd avoiding strategy it settles near
    0.8. The system has the group `diners`, the choice `choice` of a restaurant every evening, uniform or by crowd
    avoidance, and the interaction `restaurants`, which reports the fraction of restaurants that serve a meal
    `restaurants.utilization`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_agents = n_agents
        self.n_restaurants = n_restaurants
        self.strategy = strategy
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        n_restaurants = self.n_restaurants
        if self.strategy not in STRATEGIES:
            error_msg = f'Unknown strategy {self.strategy!r}. Use one of {STRATEGIES}.'
            raise ValueError(error_msg)
        n_restaurants = self.n_agents if n_restaurants is None else n_restaurants
        options = [float(restaurant) for restaurant in range(n_restaurants)]
        choices = np.random.default_rng(self.random_state).integers(n_restaurants, size=self.n_agents).astype(float)
        diners = DinerElement('diners', restaurant=choices, size=self.n_agents)
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
                schedule='synchronous',
            )
        )
        restaurants = RestaurantsInteraction('restaurants', target='diners', n_restaurants=n_restaurants)
        self.system_ = System(self.label, elements=[diners], interactions=[choice, restaurants])
