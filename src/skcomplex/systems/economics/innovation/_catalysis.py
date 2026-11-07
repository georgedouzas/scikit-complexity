"""Innovation systems: combinatorial growth of goods whose production functions catalyse each other."""

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
from ....core.components.system import System
from ..._catalog import register_system


def _combination_rate(order: int, alpha: float, goods: int) -> Callable[[SimpleNamespace], Expression]:
    """Return the rate `alpha^i C(M, i)` of new goods of `i` parents, which stops at `goods` goods."""

    def rate(namespace: SimpleNamespace) -> Expression:
        present = count(over=namespace.source)
        combinations = math.prod(present - parent for parent in range(order)) / math.factorial(order)
        return where(present < goods, alpha**order * combinations, 0)

    return rate


@register_system(
    field='economics/innovation',
    tags=('innovation', 'adjacent possible', 'autocatalytic sets', 'hypergraph', 'jump process'),
    references=('Hordijk and Kauffman 2022',),
    defaults={'until': 400},
)
def tap_catalysis(
    m0: int = 10,
    *,
    parents: int = 4,
    alpha: float = 0.01,
    goods: int = 1250,
    pc: float = 0.0044,
) -> System:
    """Return the TAP growth of goods from combinations of goods, with random catalysis of their production functions.

    The TAP model with catalysis of Hordijk and Kauffman (2022, Algorithm 1), in continuous time. From `m0` goods,
    new goods made from `i` parents drawn uniformly among the `M` goods appear at the rate `alpha^i C(M, i)`, for
    every `i` up to `parents`, until there are `goods` goods. The production function of a new good is a hyperedge of
    its parents and the good, its last member. A new good catalyses the production of every other good, and every
    other good catalyses its production, each with the probability `pc`, as directed edges from the catalyst to the
    good. Algorithm 1 draws a Poisson number of goods of every order per step from the rates at its start, and the
    process here draws them one at a time from the rates at that time. The initial goods are the food set, and
    `autocatalytic_sets` finds the largest autocatalytic set of the production functions, which exists in about
    half the runs at `pc = 0.0044` and then tends to hold most of the network.

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

    Returns:
        system:
            The system, with the goods `goods`, the catalysis `catalysis`, and the production functions `functions`.

    Examples:
        >>> from skcomplex.systems.economics.innovation import tap_catalysis
        >>> [element.label for element in tap_catalysis(goods=50).elements]
        ['goods', 'catalysis', 'functions']
    """
    growth = [
        TopologyInteraction(
            f'order_{order}',
            target='goods',
            action='add-node',
            rate=_combination_rate(order, alpha, goods),
            hyperedge=order + 1,
            independent=True,
            direction='both',
            toward=lambda node: pc,
        )
        for order in range(1, parents + 1)
    ]
    links = int(3 * pc * goods * goods) + goods
    return System(
        elements=[
            GroupElement('goods', element=Element, on='sites', variable=True, capacity=goods),
            GroupElement('catalysis', element=Element, on='edges', variable=True, capacity=links),
            GroupElement(
                'functions',
                element=Element,
                on='hyperedges',
                variable=True,
                size=0,
                capacity=goods,
                rank=parents + 1,
            ),
        ],
        interactions=growth,
        space=NetworkSpace(nx.empty_graph(m0, create_using=nx.DiGraph)),
    )
