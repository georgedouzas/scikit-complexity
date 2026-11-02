"""Econophysics models: the Kolkata Paise Restaurant problem of choosing among restaurants."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import ChoiceInteraction
from ....core.system import System
from ..._catalog import register_model
from ._crowd_avoidance import CrowdAvoidanceInteraction
from ._diner import DinerElement
from ._restaurants import RestaurantsInteraction

STRATEGIES = ('random', 'crowd_avoiding')


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
    random_state: int | np.random.Generator | None = 0,
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
            The model, with the group `diners`, the choice `choice` of a restaurant every evening, uniform or by crowd
            avoidance, and the interaction `restaurants`, which reports the fraction of
            restaurants that serve a meal `restaurants.utilization`.

    Raises:
        ValueError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.models.economics.econophysics import kolkata_restaurant
        >>> type(kolkata_restaurant(10, strategy='crowd_avoiding').interactions[0]).__name__
        'CrowdAvoidanceInteraction'
    """
    if strategy not in STRATEGIES:
        error_msg = f'Unknown strategy {strategy!r}. Use one of {STRATEGIES}.'
        raise ValueError(error_msg)
    n_restaurants = n_agents if n_restaurants is None else n_restaurants
    options = [float(restaurant) for restaurant in range(n_restaurants)]
    choices = np.random.default_rng(random_state).integers(n_restaurants, size=n_agents).astype(float)
    diners = GroupElement('diners', element=DinerElement, size=n_agents, values={'restaurant': choices})
    choice = (
        CrowdAvoidanceInteraction('choice', target='diners', options=options)
        if strategy == 'crowd_avoiding'
        else ChoiceInteraction(
            'choice',
            target='diners',
            state='restaurant',
            options=options,
            utility=lambda agent, model, option: 0.0,
            intensity=0.0,
            form='agent',
        )
    )
    restaurants = RestaurantsInteraction('restaurants', target='diners', n_restaurants=n_restaurants)
    return System(elements=[diners], interactions=[choice, restaurants])
