"""Supply chain systems: Namrata's Family.

It is the diffusion of good working conditions from buyers to suppliers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.spaces import NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import StateSpec, VisualSpec


class SupplierView(AgentView):
    """View of a firm of the supply chain diffusion system.

    Its states are attributes.
    """

    adopted: float
    inflow: float
    outflow: float


class SupplierElement(Element):
    """Firm of a supply chain, a buyer or a factory.

    It holds the value of the goods that flow into and out of it.

    Args:
        label:
            The label of the firm or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        adopted:
            Whether the firm has adopted good working conditions.

        inflow:
            The total value `w_->i` of the goods that flow into the firm from its suppliers.

        outflow:
            The total value `w_j->` of the goods that flow out of the firm to its buyers.

    Examples:
        >>> from skcomplex.systems.social.organizations import SupplierElement
        >>> SupplierElement('factory', outflow=6.0).outflow
        6.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'adopted', 'color_range': (0.0, 1.0)}),
        *,
        adopted: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        inflow: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        outflow: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.adopted = adopted
        self.inflow = inflow
        self.outflow = outflow

    @classmethod
    def _get_view(cls) -> type[SupplierView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return SupplierView


class SupplyChainDiffusionSystem(BaseSystem):
    """Chesney's Namrata's Family.

    Buyers push good working conditions down their supply network. Goods flow from suppliers to buyers along the
    weighted edges of a directed network, `w_ji` from `j` to `i`. A buyer `i` that has adopted forces a supplier `j` to
    adopt when it is less beholden to the supplier than the supplier is to it,
    `V_ij = w_ji / w_->i < V_ji = w_ji / w_j->`, where `w_->i` is the total value that flows into `i` and `w_j->` the
    total value that flows out of `j`, with a strict inequality since factories resist by default (Chesney 2021 sections
    4.9 and 4.10). A seed group of buyers adopts at the start, and in every time period the firms that adopted in the
    previous one push their suppliers. Since the condition reads `w_->i > w_j->`, a buyer with a single supplier never
    forces it, so diffusion needs buyers with two or more suppliers (section 4.11). The default network has `buyers`
    buyers and `factories` factories, every factory supplying one to three random firms listed before it with values
    from 1 to 10.

    The system has the group `firms` on the nodes of the network.

    Args:
        graph:
            A NetworkX directed graph of the supply network with the values of the goods as the edge attribute
            `weight`, or `None` for a random network.

        buyers:
            The number of buyers, the first nodes, of the default network.

        factories:
            The number of factories of the default network.

        seeds:
            The number of the first nodes that adopt at the start.

        random_state:
            The seed or generator of the default network.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import SupplyChainDiffusionSystem
        >>> SupplyChainDiffusionSystem(factories=10).prepare().system_.elements[0].prepare().size_
        13
    """

    field = 'social/organizations'
    topics = ('agents', 'network', 'supply chains', 'diffusion', 'labour standards')
    references = ('Chesney 2021', 'Gold, Chesney, Gruchmann, and Trautrims 2020')
    until = 10

    def __init__(
        self,
        graph: object = None,
        *,
        buyers: int = 3,
        factories: int = 40,
        seeds: int = 2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.buyers = buyers
        self.factories = factories
        self.seeds = seeds
        self.random_state = random_state

    @staticmethod
    def _adopt(firm: SupplierView, system: SystemViewProtocol) -> float:
        """Return 1 for a firm that has adopted or can be forced, and 0 otherwise.

        A firm can be forced when an adopting buyer has the power to force it. A buyer `i` forces its supplier `j` when
        `V_ij = w_ji / w_->i` is below `V_ji = w_ji / w_j->`, that is when the inflow of the buyer exceeds the outflow
        of the supplier.
        """
        del system
        if firm.adopted:
            return 1.0
        buyers = [item for item in firm.neighbors() if isinstance(item, SupplierView)]
        return float(any(buyer.adopted and buyer.inflow > firm.outflow for buyer in buyers))

    @staticmethod
    def _supply_network(buyers: int, factories: int, rng: np.random.Generator) -> nx.DiGraph:
        """Return a random supply network.

        Its edges go from supplier to buyer, weighted by the value of the goods.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(range(buyers + factories))
        for factory in range(buyers, buyers + factories):
            clients = rng.choice(factory, min(factory, int(rng.integers(1, 4))), replace=False)
            for client in clients.tolist():
                graph.add_edge(factory, client, weight=float(rng.integers(1, 11)))
        return graph

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        network = self._supply_network(self.buyers, self.factories, rng) if self.graph is None else self.graph
        if not isinstance(network, nx.DiGraph):
            network = nx.DiGraph(network)
        nodes = list(network.nodes)
        adopted = np.zeros(len(nodes))
        adopted[: self.seeds] = 1.0
        firms = GroupElement(
            'firms',
            element=SupplierElement,
            size=len(nodes),
            values={
                'adopted': adopted,
                'inflow': [float(network.in_degree(node, weight='weight')) for node in nodes],
                'outflow': [float(network.out_degree(node, weight='weight')) for node in nodes],
            },
        )
        adoption = MapInteraction('adoption', target='firms', state='adopted', function=self._adopt, form='agent')
        return CompositeSystem(self.label, elements=[firms], interactions=[adoption], space=NetworkSpace(network))
