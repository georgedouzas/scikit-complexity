"""Ising spins on the nodes of a graph, sampled by colored Monte Carlo sweeps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import AdjacencyNeighborhood, NetworkSpace
from ....core.components.systems import BaseSystem, System
from ..common.spins import IsingSamplingInteraction, IsingSpinElement

NODES = 400
DEGREE = 4


class GraphIsingSystem(BaseSystem):
    """The Ising system on a graph.

    The system orders below a critical temperature set by the degrees of the nodes.

    On a random 4-regular graph the Bethe approximation gives the critical temperature `2 / ln 2`. One unit of time is
    one sweep of colored updates. The system has the spins `spins.spin` in node order.

    Args:
        graph:
            The graph whose nodes hold the spins, or `None` for a random 4-regular graph of 400 nodes.

        temperature:
            The temperature in units of the coupling, which can change during a run.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        initial:
            `'random'` for random spins or `'up'` for all spins up.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        random_state:
            The seed or generator of the default graph and the random initial spins.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.physics.statistical_mechanics import GraphIsingSystem
        >>> GraphIsingSystem(nx.cycle_graph(10)).prepare().system_.elements[0].prepare().size_
        10
    """

    field = "physics/statistical_mechanics"
    topics = ("network", "Monte Carlo", "phase transition", "spins")
    references = ("Argun et al. 2021", "Dorogovtsev, Goltsev, and Mendes 2008")
    until = 200
    dt = 1

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        temperature: float = 2.0,
        J: float = 1.0,
        h: float = 0.0,
        initial: str = "random",
        sampler: str = "metropolis",
        random_state: int | np.random.Generator | None = 0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.temperature = temperature
        self.J = J
        self.h = h
        self.initial = initial
        self.sampler = sampler
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the Ising system on a graph.

        The system orders below a critical temperature set by the degrees of the nodes.
        """
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        if graph is None:
            graph = nx.random_regular_graph(DEGREE, NODES, seed=int(rng.integers(2**31)))
        size = graph.number_of_nodes()
        spins = np.ones(size) if self.initial == "up" else rng.choice([-1.0, 1.0], size=size)
        space = NetworkSpace(graph)
        group = GroupElement(
            "spins",
            element=IsingSpinElement,
            space=space,
            values={"spin": spins, "J": self.J, "h": self.h, "temperature": self.temperature},
            on="sites",
        )
        sampling = IsingSamplingInteraction(
            "sampling",
            target="spins",
            rule=self.sampler,
            schedule="colored",
            neighborhood=AdjacencyNeighborhood(),
        )
        return System(self.label, elements=[group], interactions=[sampling])
