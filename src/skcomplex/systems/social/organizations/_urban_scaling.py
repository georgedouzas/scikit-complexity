"""The gravity system of urban scaling, pair interactions summed over a city."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.base import BaseComponent
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import EquationLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import ContinuousSpace, RadiusNeighborhood, StaticTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec


class CityResidentElement(Element):
    """Resident of a city, who stays at a position.

    Args:
        label:
            The label of the resident or the group.

        x:
            The position.

        contacts:
            The initial guess of the sum of the interactions with the others, which the equation replaces.

    Examples:
        >>> from skcomplex.systems.social.organizations import CityResidentElement
        >>> CityResidentElement('resident', x=(1.0, 2.0)).x
        (1.0, 2.0)
    """

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        contacts: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.contacts = contacts


class GravityContactInteraction(Interaction):
    """Contacts of every resident.

    They are the sum of `r^-gamma` over the others within a reach.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of residents.

        reach:
            The largest distance of a contact, which the diameter of a city covers.

        gamma:
            The decay exponent of an interaction with the distance.

    Examples:
        >>> from skcomplex.systems.social.organizations import GravityContactInteraction
        >>> GravityContactInteraction('contact', target='residents', reach=30.0, gamma=1.5).gamma
        1.5
    """

    law_classes = (EquationLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        reach: float,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.reach = reach
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> Sequence[BaseComponent]:
        """Return the equation of the contacts of a resident.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The equation of the contacts.
        """
        dx, dy = s.target.other.x[0] - s.target.x[0], s.target.other.x[1] - s.target.x[1]
        contacts = neighbors((dx**2 + dy**2) ** (-s.gamma / 2), over=RadiusNeighborhood(self.reach))
        return [EquationLaw(s.target.contacts == contacts, target=s.target.contacts)]


class GravityUrbanScalingSystem(BaseSystem):
    """The residents of a city at uniform positions in a disc.

    Each resident interacts with the others as `r^-gamma`. The disc has the area `N^alpha`, so the total interaction of
    a city grows as `N^(2 - alpha gamma / 2)` for `gamma < 2`, which with the population dimension `D_P = 2 / alpha` is
    the exponent `2 - gamma / D_P` of Ribeiro and Netto (2024 §4). A `sweep` varies `population` and fits the exponent
    with `scaling(kind='power')`. The system has the states `residents.x` and `residents.contacts` and the observable
    `total_contacts` of the catalog, which a run passes in `observables`.

    Args:
        population:
            The number of residents `N`.

        alpha:
            The exponent of the area of the city with its population.

        gamma:
            The decay exponent of an interaction with the distance.

        random_state:
            The seed or generator of the positions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import GravityUrbanScalingSystem
        >>> GravityUrbanScalingSystem(100).prepare().system_.elements[0].size
        100
    """

    field = 'social/organizations'
    topics = ('urban scaling', 'gravity system', 'static', 'allometry')
    references = ('Ribeiro and Netto 2024, arXiv 2404.02642', 'Ribeiro et al. 2017')
    observables = (ObservableSpec('total_contacts', group='residents', variable='contacts', reduce='sum'),)

    def __init__(
        self,
        population: int = 400,
        *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.population = population
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        radius = np.sqrt(self.population**self.alpha / np.pi)
        rng = np.random.default_rng(self.random_state)
        distances = radius * np.sqrt(rng.uniform(size=self.population))
        angles = rng.uniform(0.0, 2 * np.pi, self.population)
        positions = radius + distances[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
        residents = GroupElement(
            'residents',
            element=CityResidentElement,
            size=self.population,
            values={'x': positions},
        )
        contact = GravityContactInteraction('contact', target='residents', reach=2.0 * radius, gamma=self.gamma)
        space = ContinuousSpace(extent=[2.0 * radius, 2.0 * radius])
        return CompositeSystem(self.label, elements=[residents], interactions=[contact], space=space, time=StaticTime())
