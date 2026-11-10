"""Urban scaling of interactions: a city grown by hyperbolic distance.

Its residents meet their neighbours.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import exp, ln, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction, TopologyInteraction
from ....core.components.laws.utils import draw, logistic, neighbors
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, StateSpec, VisualSpec

MEETINGS_VISUAL = VisualSpec('sites', {'states': ['meetings']})
SMALLEST_ANGLE = 1e-12
MIN_EXPONENT = 2.0


class CityDwellerElement(Element):
    """Resident of a growing city.

    It holds an angle of similarity, an age, and the meetings of the last step.

    Args:
        label:
            The label of the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        angle:
            The angle of the resident on the circle, its similarity coordinate.

        age:
            The number of steps since the resident joined.

        degree:
            The number of neighbours of the resident at the start of the last step.

        meetings:
            The number of neighbours the resident met in the last step.

    Examples:
        >>> from skcomplex.systems.social.organizations import CityDwellerElement
        >>> CityDwellerElement('residents', angle=1.5).angle
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = MEETINGS_VISUAL,
        *,
        angle: Annotated[float, StateSpec()] = 0.0,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        meetings: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.angle = angle
        self.age = age
        self.degree = degree
        self.meetings = meetings


class UrbanScalingInteractionsSystem(BaseSystem):
    """A city that grows by residents and internal links.

    Its meetings grow faster than its population. Every step a resident joins at a uniform angle and adds `edges` links
    to residents drawn in proportion to the Fermi-Dirac probability of their hyperbolic distance, as in
    popularity-similarity growth (Papadopoulos et al. 2012). Every resident also adds an internal link with the
    probability `internal` per step, toward the same weight, so the links per step and the mean degree grow with the
    size. This rate per resident is the assumption of this system, since Papadopoulos et al. (2015) add a fixed number
    of internal links per step and keep the mean degree constant. Every resident meets each neighbour with the
    probability `meeting` per step. The growth and the internal links stop when the city holds `population` residents,
    so one run time reaches every size. A `sweep` varies `population`, and `scaling(kind='allometric')` fits the
    exponent of the meetings, which Ribeiro and Netto (2024) review near 1.15 for socioeconomic quantities, and which
    the rate `internal` sets here.

    The system has the groups `residents` and `links`, the states `residents.angle`, `residents.age`,
    `residents.degree`, and `residents.meetings`, and the observable `total_meetings` of the catalog, which a run passes
    in `observables`.

    Args:
        population:
            The number of residents the city holds room for.

        edges:
            The number of links of every new resident, and one less than the residents of the complete seed city.

        internal:
            The probability per step that a resident adds an internal link.

        meeting:
            The probability per step that a resident meets one neighbour.

        exponent:
            The exponent of the distribution of the links of the growth without internal links, above 2.

        temperature:
            The temperature of the network, between 0 and 1.

        random_state:
            The seed or generator of the angles of the seed city.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The exponent is not above 2, or the temperature is not between 0 and 1.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.social.organizations import UrbanScalingInteractionsSystem
        >>> trajectory = simulate(UrbanScalingInteractionsSystem(13, internal=0.0), until=10, random_state=0)
        >>> int(trajectory.get_network(-1).sources.size)
        23
    """

    field = 'social/organizations'
    topics = ('urban scaling', 'network', 'growth', 'hyperbolic', 'allometry')
    references = (
        'Ribeiro and Netto 2024, arXiv 2404.02642',
        'Papadopoulos et al. 2012, arXiv 1106.0286',
        'Papadopoulos et al. 2015',
    )
    until = 297
    observables = (ObservableSpec('total_meetings', group='residents', variable='meetings', reduce='sum'),)

    def __init__(
        self,
        population: int = 300,
        *,
        edges: int = 2,
        internal: float = 0.002,
        meeting: float = 0.5,
        exponent: float = 2.5,
        temperature: float = 0.2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.population = population
        self.edges = edges
        self.internal = internal
        self.meeting = meeting
        self.exponent = exponent
        self.temperature = temperature
        self.random_state = random_state

    @staticmethod
    def _get_closeness(node: SimpleNamespace, arrivals: int, edges: int, fading: float, temperature: float) -> object:
        """Return the Fermi-Dirac probability of the hyperbolic distance of a candidate.

        The distance is from the source of an edge.
        """
        now = node.t + arrivals
        radius = 2 * ln(now)
        first = fading * 2 * ln(now - node.age) + (1 - fading) * radius
        second = fading * 2 * ln(now - node.source.age) + (1 - fading) * radius
        angle = pi - abs(pi - abs(node.angle - node.source.angle))
        distance = first + second + 2 * ln(angle / 2 + SMALLEST_ANGLE)
        spread = (
            2 * temperature * (1 - exp(-(1 - fading) * radius / 2)) / (sin(temperature * pi) * edges * (1 - fading))
        )
        return 1 - logistic(distance, radius - 2 * ln(spread), 2 * temperature)

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        if self.exponent <= MIN_EXPONENT:
            error_msg = f'The degree exponent of the city is {self.exponent!r}. Give an exponent above {MIN_EXPONENT}.'
            raise ValidationError(error_msg)
        if not 0 < self.temperature < 1:
            error_msg = f'The temperature of the city is {self.temperature!r}. Give a temperature between 0 and 1.'
            raise ValidationError(error_msg)
        seed_nodes = self.edges + 1
        fading = 1 / (self.exponent - 1)
        arrivals = seed_nodes + 1

        def closeness(node: SimpleNamespace) -> object:
            return self._get_closeness(node, arrivals, self.edges, fading, self.temperature)

        angles = np.random.default_rng(self.random_state).uniform(0.0, 2 * np.pi, seed_nodes)
        ages = np.arange(seed_nodes - 1, -1, -1, dtype=float)
        residents = GroupElement(
            'residents',
            element=CityDwellerElement,
            on='sites',
            variable=True,
            capacity=self.population,
            values={'age': ages.tolist(), 'angle': angles.tolist()},
        )
        capacity = self.edges * self.population + seed_nodes**2 + int(np.ceil(self.internal * self.population**2))
        links = GroupElement('links', element=Element, on='edges', variable=True, capacity=capacity)
        growth = TopologyInteraction(
            'growth',
            action='add-node',
            edges=self.edges,
            condition=lambda s: s.t < self.population - seed_nodes,
            values={'angle': lambda rng: rng.uniform(0.0, 2 * np.pi)},
            toward=closeness,
        )
        bridging = TopologyInteraction(
            'bridging',
            target='residents',
            action='add-edge',
            probability=self.internal,
            condition=lambda s: s.t < self.population - seed_nodes,
            toward=closeness,
        )
        ageing = MapInteraction('ageing', target='residents', state='age', function=lambda s: 1, mode='increment')
        degree = MapInteraction('degree', target='residents', state='degree', function=lambda s: neighbors(1))
        meetings = MapInteraction(
            'meetings',
            target='residents',
            state='meetings',
            function=lambda s: draw('meet', 'binom', n=s.target.degree, p=self.meeting),
        )
        return CompositeSystem(
            self.label,
            elements=[residents, links],
            interactions=[growth, bridging, ageing, degree, meetings],
            space=NetworkSpace(nx.complete_graph(seed_nodes)),
            time=DiscreteTime(),
        )
