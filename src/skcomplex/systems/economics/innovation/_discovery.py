"""Innovation systems: discovery processes of explorers whose urns grow and share their adjacent possible."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx

from ....core.components.elements import GroupElement
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.specs import PhaseSpec, PhasesScheduleSpec
from ..._catalog import register_system
from ._ball import BallElement
from ._urn import DRAW, GROW, UrnElement


@register_system(
    field='economics/innovation',
    tags=('innovation', 'adjacent possible', 'urn', 'network', 'agents', 'heaps law'),
    references=('Iacopini et al. 2020',),
    defaults={'until': 100},
)
def interacting_urns_network(
    graph: nx.Graph | None = None,
    *,
    rho: int = 4,
    nu: int = 1,
    m0: int | None = None,
) -> System:
    """Return one urn with triggering per node of a network, which draws from its own urn and those of its neighbours.

    The discovery process of Iacopini et al. (2020, Eqs. 1 and 2). Every node holds an urn of `m0` balls of colours
    that no other urn holds. Every step every node draws one ball from its social urn, its own urn and one ball of every
    colour of the urns of its neighbours, all from the urns at the start of the step. It then adds `rho` balls of the
    drawn colour to its own urn, and when it draws a colour for the first time, a novelty, `nu + 1` balls of colours
    that no urn held before. The number of novelties of a node grows as Heaps' law `D_i(t) ~ t^beta_i`, with
    `beta = nu / rho` for an isolated urn and higher exponents for nodes whose neighbours enlarge their social urns.

    The urn of a node is the set of ball agents of the variable group `balls` on its node, one agent per colour with
    its multiplicity, which the node reads with `agents_here` and its neighbours with `neighbors(group='balls')`.

    Args:
        graph:
            The NetworkX graph of the explorers, or `None` for Zachary's karate club.

        rho:
            The reinforcement, the balls of the drawn colour added after every draw.

        nu:
            The triggering, one less than the colours added after a novelty.

        m0:
            The initial number of balls of every urn, or `None` for `nu + 1`.

    Returns:
        system:
            The system, with the urns `urns` and their novelties `urns.novelties`, and the balls `balls`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.economics.innovation import interacting_urns_network
        >>> system = interacting_urns_network(nx.path_graph(3), rho=2, nu=1)
        >>> [element.prepare().size_ for element in system.elements]
        [3, 6]
    """
    graph = nx.karate_club_graph() if graph is None else graph
    nodes = list(graph.nodes)
    initial = nu + 1 if m0 is None else m0
    places = [node for node in nodes for _ in range(initial)]
    colors = list(range(len(places)))
    return System(
        elements=[
            GroupElement('urns', element=UrnElement, size=len(nodes), location=nodes, values={'rho': rho, 'nu': nu}),
            GroupElement(
                'balls',
                element=BallElement,
                size=len(places),
                variable=True,
                location=places,
                values={'color': colors},
            ),
        ],
        space=NetworkSpace(graph),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec((PhaseSpec(DRAW, activation='simultaneous'), PhaseSpec(GROW))),
        ),
    )
