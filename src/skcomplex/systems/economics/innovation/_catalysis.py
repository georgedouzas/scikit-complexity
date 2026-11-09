"""Innovation systems: combinatorial growth of goods.

Their production functions catalyse each other.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from types import SimpleNamespace

import networkx as nx
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TopologyInteraction
from ....core.components.laws.utils import count, where
from ....core.components.spaces import NetworkSpace
from ....core.components.systems import BaseSystem, System


class TapCatalysisSystem(BaseSystem):
    """The TAP growth of goods from combinations of goods.

    Their production functions are catalysed at random. The TAP model with catalysis of Hordijk and Kauffman (2022,
    Algorithm 1), in continuous time. From `m0` goods, new goods made from `i` parents drawn uniformly among the `M`
    goods appear at the rate `alpha^i C(M, i)`, for every `i` up to `parents`, until there are `goods` goods. The
    production function of a new good is a hyperedge of its parents and the good, its last member. A new good catalyses
    the production of every other good, and every other good catalyses its production, each with the probability `pc`,
    as directed edges from the catalyst to the good. Algorithm 1 draws a Poisson number of goods of every order per step
    from the rates at its start, and the process here draws them one at a time from the rates at that time. The initial
    goods are the food set, and `autocatalytic_sets` finds the largest autocatalytic set of the production functions,
    which exists in about half the runs at `pc = 0.0044` and then tends to hold most of the network.

    The system has the goods `goods`, the catalysis `catalysis`, and the production functions `functions`.

    Args:
        m0:
            The number of initial goods, the food set.

        parents:
            The largest number of parents of a good, `K`.

        alpha:
            The base of the probabilities `alpha^i` that a combination of `i` goods makes a new one.

        goods:
            The number of goods at which the growth stops, `M`.

        pc:
            The catalysis probability.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.innovation import TapCatalysisSystem
        >>> [element.label for element in TapCatalysisSystem(goods=50).prepare().system_.elements]
        ['goods', 'catalysis', 'functions']
    """

    field = 'economics/innovation'
    topics = ('innovation', 'adjacent possible', 'autocatalytic sets', 'hypergraph', 'jump process')
    references = ('Hordijk and Kauffman 2022',)
    until = 400

    def __init__(
        self,
        m0: int = 10,
        *,
        parents: int = 4,
        alpha: float = 0.01,
        goods: int = 1250,
        pc: float = 0.0044,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.m0 = m0
        self.parents = parents
        self.alpha = alpha
        self.goods = goods
        self.pc = pc

    @staticmethod
    def _combination_rate(order: int, alpha: float, goods: int) -> Callable[[SimpleNamespace], Expression]:
        """Return the rate `alpha^i C(M, i)` of new goods of `i` parents.

        It stops at `goods` goods.
        """

        def rate(namespace: SimpleNamespace) -> Expression:
            present = count(over=namespace.source)
            combinations = math.prod(present - parent for parent in range(order)) / math.factorial(order)
            return where(present < goods, alpha**order * combinations, 0)

        return rate

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        growth = [
            TopologyInteraction(
                f'order_{order}',
                target='goods',
                action='add-node',
                rate=self._combination_rate(order, self.alpha, self.goods),
                hyperedge=order + 1,
                independent=True,
                direction='both',
                toward=lambda node: self.pc,
            )
            for order in range(1, self.parents + 1)
        ]
        links = int(3 * self.pc * self.goods * self.goods) + self.goods
        return System(
            self.label,
            elements=[
                GroupElement('goods', element=Element, on='sites', variable=True, capacity=self.goods),
                GroupElement('catalysis', element=Element, on='edges', variable=True, capacity=links),
                GroupElement(
                    'functions',
                    element=Element,
                    on='hyperedges',
                    variable=True,
                    size=0,
                    capacity=self.goods,
                    rank=self.parents + 1,
                ),
            ],
            interactions=growth,
            space=NetworkSpace(nx.empty_graph(self.m0, create_using=nx.DiGraph)),
        )
