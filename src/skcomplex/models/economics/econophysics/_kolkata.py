"""Econophysics models: the Kolkata Paise Restaurant problem of choosing among restaurants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.base import Domain, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

STRATEGIES = ('random', 'crowd_avoiding')


class Diner(AgentType):
    """Diner of the Kolkata Paise Restaurant problem, who picks a restaurant at random every evening.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        restaurant:
            The restaurant chosen last, from 0 to the number of restaurants minus one.

        crowd:
            The number of diners at that restaurant, 0 before the first evening.

        served:
            Whether the diner got a meal, 1 or 0.

        restaurants:
            The number of restaurants.

    Examples:
        >>> from skcomplex.models.economics.econophysics import Diner
        >>> Diner.group('diners', 10, restaurants=10.0).size
        10
    """

    restaurant: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    crowd: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    served: float = state(default=0.0, domain=Domain.BOOLEAN)
    restaurants: float = parameter(default=100.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the choice rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.choose)]

    def choose(self, agent: AgentView, model: ModelView) -> None:
        """Choose a restaurant uniformly at random.

        Args:
            agent:
                The diner.

            model:
                The model.
        """
        agent.restaurant = float(model.rng.integers(int(agent.restaurants)))


class CrowdAvoidingDiner(Diner):
    """Diner of the stochastic crowd avoiding strategy, who leaves crowded restaurants.

    A diner returns to last evening's restaurant with probability `1 / n`, with `n` the crowd it found there, and
    otherwise picks one of the other restaurants at random, which raises the utilization to about 0.8.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        restaurant:
            The restaurant chosen last, from 0 to the number of restaurants minus one.

        crowd:
            The number of diners at that restaurant, 0 before the first evening.

        served:
            Whether the diner got a meal, 1 or 0.

        restaurants:
            The number of restaurants.

    Examples:
        >>> from skcomplex.models.economics.econophysics import CrowdAvoidingDiner
        >>> CrowdAvoidingDiner.group('diners', 10, restaurants=10.0).size
        10
    """

    def choose(self, agent: AgentView, model: ModelView) -> None:
        """Stay with probability one over the crowd, otherwise move to another random restaurant.

        Args:
            agent:
                The diner.

            model:
                The model.
        """
        if agent.crowd < 1:
            super().choose(agent, model)
        elif model.rng.random() >= 1 / agent.crowd:
            other = int(model.rng.integers(int(agent.restaurants) - 1))
            agent.restaurant = float(other + (other >= agent.restaurant))


class Restaurants(Interaction):
    """Restaurants that each serve one of the diners who come, chosen at random, every evening.

    Args:
        label:
            The label of the interaction.

        diners:
            The path of the group of diners.

        n_restaurants:
            The number of restaurants.

    Examples:
        >>> from skcomplex.models.economics.econophysics import Restaurants
        >>> Restaurants('restaurants', diners='diners', n_restaurants=5).n_restaurants
        5
    """

    reports: ClassVar[tuple[str, ...]] = ('utilization',)
    diners: str = role()
    n_restaurants: int = 100

    def laws(self, s: Symbols) -> list[Law]:
        """Return the serving rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.serve)]

    def serve(self, model: ModelView) -> None:
        """Serve one random diner per visited restaurant and report the fraction of restaurants that serve.

        Args:
            model:
                The model.
        """
        agents = model.agents('diners')
        choices = np.array([agent.restaurant for agent in agents], dtype=np.int64)
        crowds = np.bincount(choices, minlength=self.n_restaurants)
        order = model.rng.permutation(len(agents))
        _, first = np.unique(choices[order], return_index=True)
        served = np.zeros(len(agents), dtype=bool)
        served[order[first]] = True
        for agent, choice, meal in zip(agents, choices, served, strict=True):
            agent.crowd, agent.served = float(crowds[choice]), float(meal)
        model.report('utilization', np.count_nonzero(crowds) / self.n_restaurants)


@register_model(
    field='economics/econophysics',
    tags=('kolkata paise restaurant', 'agents', 'resource allocation', 'minority game'),
    references=('Chakrabarti et al. 2009', 'Ghosh et al. 2010', 'Carmichael and Hadzikadic 2019 ch. 9'),
    defaults={'until': 100},
)
def kolkata_restaurant(
    n_agents: int = 100,
    *,
    n_restaurants: int | None = None,
    strategy: str = 'random',
    random_state: int | np.random.Generator | None = None,
) -> System:
    """Return the Kolkata Paise Restaurant problem, where diners spread over restaurants that serve one meal each.

    With random choice the expected utilization of `N` restaurants by `N` diners is `1 - 1/e`, about 0.632, and with
    the stochastic crowd avoiding strategy it settles near 0.8.

    Args:
        n_agents:
            The number of diners.

        n_restaurants:
            The number of restaurants, or `None` for as many as diners.

        strategy:
            `'random'` or `'crowd_avoiding'`.

        random_state:
            The seed or generator of the first choices.

    Returns:
        system:
            The model, with the group `diners` and the interaction `restaurants`, which reports the fraction of
            restaurants that serve a meal `restaurants.utilization`.

    Raises:
        ValueError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.models.economics.econophysics import kolkata_restaurant
        >>> kolkata_restaurant(10, strategy='crowd_avoiding').elements[0].element.__name__
        'CrowdAvoidingDiner'
    """
    if strategy not in STRATEGIES:
        error_msg = f'Unknown strategy {strategy!r}. Use one of {STRATEGIES}.'
        raise ValueError(error_msg)
    n_restaurants = n_agents if n_restaurants is None else n_restaurants
    diner = CrowdAvoidingDiner if strategy == 'crowd_avoiding' else Diner
    choices = np.random.default_rng(random_state).integers(n_restaurants, size=n_agents).astype(float)
    diners = diner.group('diners', n_agents, restaurant=choices, restaurants=float(n_restaurants))
    restaurants = Restaurants('restaurants', diners='diners', n_restaurants=n_restaurants)
    return System(elements=[diners], interactions=[restaurants])
