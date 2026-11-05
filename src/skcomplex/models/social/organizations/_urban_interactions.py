"""Urban scaling of interactions: a city grown by hyperbolic distance, whose residents meet their neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import exp, ln, pi, sin

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction, TopologyInteraction
from ....core.laws.utils import draw, logistic, neighbors
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....errors import ValidationError
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._city_dweller import CityDwellerElement

SMALLEST_ANGLE = 1e-12
MIN_EXPONENT = 2.0


def _get_closeness(node: SimpleNamespace, arrivals: int, edges: int, fading: float, temperature: float) -> object:
    """Return the Fermi-Dirac probability of the hyperbolic distance between a candidate and the source of an edge."""
    now = node.t + arrivals
    radius = 2 * ln(now)
    first = fading * 2 * ln(now - node.age) + (1 - fading) * radius
    second = fading * 2 * ln(now - node.source.age) + (1 - fading) * radius
    angle = pi - abs(pi - abs(node.angle - node.source.angle))
    distance = first + second + 2 * ln(angle / 2 + SMALLEST_ANGLE)
    spread = 2 * temperature * (1 - exp(-(1 - fading) * radius / 2)) / (sin(temperature * pi) * edges * (1 - fading))
    return 1 - logistic(distance, radius - 2 * ln(spread), 2 * temperature)


def _total_meetings(system: System) -> list[ObservableSpec]:
    """Return the number of meetings of the city in the last step, each counted from both ends."""
    return [ObservableSpec('total_meetings', group='residents', variable='meetings', reduce='sum')]


@register_model(
    field='social/organizations',
    tags=('urban scaling', 'network', 'growth', 'hyperbolic', 'allometry'),
    references=(
        'Ribeiro and Netto 2024, arXiv 2404.02642',
        'Papadopoulos et al. 2012, arXiv 1106.0286',
        'Papadopoulos et al. 2015',
    ),
    observables=_total_meetings,
    defaults={'until': 297},
)
def urban_scaling_interactions(
    population: int = 300,
    *,
    edges: int = 2,
    internal: float = 0.002,
    meeting: float = 0.5,
    exponent: float = 2.5,
    temperature: float = 0.2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a city that grows by residents and internal links, so its meetings grow faster than its population.

    Every step a resident joins at a uniform angle and adds `edges` links to residents drawn in proportion to the
    Fermi-Dirac probability of their hyperbolic distance, as in popularity-similarity growth (Papadopoulos et al.
    2012). Every resident also adds an internal link with the probability `internal` per step, toward the same weight,
    so the links per step and the mean degree grow with the size. This rate per resident is the assumption of this
    model, since Papadopoulos et al. (2015) add a fixed number of internal links per step and keep the mean degree
    constant.
    Every resident meets each neighbour with the probability `meeting` per step. The growth and the internal links stop
    when the city holds `population` residents, so one run time reaches every size. An `Experiment` sweeps `population`,
    and `ScalingAnalysis(kind='allometric')` fits the exponent of the meetings, which Ribeiro and Netto (2024) review
    near 1.15 for socioeconomic quantities, and which the rate `internal` sets here.

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

    Returns:
        system:
            The model, with the groups `residents` and `links`, the states `residents.angle`, `residents.age`,
            `residents.degree`, and `residents.meetings`, and the observable `total_meetings` of the catalog, which a
            run passes in `observables`.

    Raises:
        ValidationError:
            The exponent is not above 2, or the temperature is not between 0 and 1.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.social.organizations import urban_scaling_interactions
        >>> system = urban_scaling_interactions(13, internal=0.0)
        >>> trajectory = Simulation(system, random_state=0).prepare().run(until=10).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        23
    """
    if exponent <= MIN_EXPONENT:
        error_msg = f'The degree exponent of the city is {exponent!r}. Give an exponent above {MIN_EXPONENT}.'
        raise ValidationError(error_msg)
    if not 0 < temperature < 1:
        error_msg = f'The temperature of the city is {temperature!r}. Give a temperature between 0 and 1.'
        raise ValidationError(error_msg)
    seed_nodes = edges + 1
    fading = 1 / (exponent - 1)
    arrivals = seed_nodes + 1

    def closeness(node: SimpleNamespace) -> object:
        return _get_closeness(node, arrivals, edges, fading, temperature)

    angles = np.random.default_rng(random_state).uniform(0.0, 2 * np.pi, seed_nodes)
    ages = np.arange(seed_nodes - 1, -1, -1, dtype=float)
    residents = GroupElement(
        'residents',
        element=CityDwellerElement,
        on='sites',
        variable=True,
        capacity=population,
        values={'age': ages.tolist(), 'angle': angles.tolist()},
    )
    capacity = edges * population + seed_nodes**2 + int(np.ceil(internal * population**2))
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=capacity)
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        edges=edges,
        condition=lambda s: s.t < population - seed_nodes,
        values={'angle': lambda rng: rng.uniform(0.0, 2 * np.pi)},
        toward=closeness,
    )
    bridging = TopologyInteraction(
        'bridging',
        target='residents',
        action='add-edge',
        probability=internal,
        condition=lambda s: s.t < population - seed_nodes,
        toward=closeness,
    )
    ageing = MapInteraction('ageing', target='residents', state='age', function=lambda s: 1, mode='increment')
    degree = MapInteraction('degree', target='residents', state='degree', function=lambda s: neighbors(1))
    meetings = MapInteraction(
        'meetings',
        target='residents',
        state='meetings',
        function=lambda s: draw('meet', 'binom', n=s.target.degree, p=meeting),
    )
    return System(
        elements=[residents, links],
        interactions=[growth, bridging, ageing, degree, meetings],
        space=NetworkSpace(nx.complete_graph(seed_nodes)),
        time=DiscreteTime(),
    )
