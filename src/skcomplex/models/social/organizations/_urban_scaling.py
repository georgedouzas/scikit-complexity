"""The gravity model of urban scaling, pair interactions summed over a city."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.elements import GroupElement
from ....core.spaces import ContinuousSpace, StaticTime
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._city_resident import CityResidentElement
from ._gravity_contact import GravityContactInteraction


def _total_contacts(system: System) -> list[ObservableSpec]:
    """Return the total interaction of the city."""
    return [ObservableSpec('total_contacts', group='residents', variable='contacts', reduce='sum')]


@register_model(
    field='social/organizations',
    tags=('urban scaling', 'gravity model', 'static', 'allometry'),
    references=('Ribeiro and Netto 2024, arXiv 2404.02642', 'Ribeiro et al. 2017'),
    observables=_total_contacts,
)
def gravity_urban_scaling(
    population: int = 400,
    *,
    alpha: float = 1.0,
    gamma: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the residents of a city at uniform positions in a disc, each interacting with the others as `r^-gamma`.

    The disc has the area `N^alpha`, so the total interaction of a city grows as `N^(2 - alpha gamma / 2)` for
    `gamma < 2`, which with the population dimension `D_P = 2 / alpha` is the exponent `2 - gamma / D_P` of Ribeiro and
    Netto (2024 §4). An `Experiment` sweeps `population` and fits the exponent with `ScalingAnalysis(kind='power')`.

    Args:
        population:
            The number of residents `N`.

        alpha:
            The exponent of the area of the city with its population.

        gamma:
            The decay exponent of an interaction with the distance.

        random_state:
            The seed or generator of the positions.

    Returns:
        system:
            The model, with the states `residents.x` and `residents.contacts` and the observable `total_contacts` of
            the catalog, which a run passes in `observables`.

    Examples:
        >>> from skcomplex.models.social.organizations import gravity_urban_scaling
        >>> gravity_urban_scaling(100).elements[0].size
        100
    """
    radius = np.sqrt(population**alpha / np.pi)
    rng = np.random.default_rng(random_state)
    distances = radius * np.sqrt(rng.uniform(size=population))
    angles = rng.uniform(0.0, 2 * np.pi, population)
    positions = radius + distances[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    residents = GroupElement('residents', element=CityResidentElement, size=population, values={'x': positions})
    contact = GravityContactInteraction('contact', target='residents', reach=2.0 * radius, gamma=gamma)
    space = ContinuousSpace(extent=[2.0 * radius, 2.0 * radius])
    return System(elements=[residents], interactions=[contact], space=space, time=StaticTime())
