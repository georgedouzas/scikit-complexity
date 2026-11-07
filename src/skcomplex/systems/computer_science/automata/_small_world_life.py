"""Automata systems on networks: the Game of Life on a small-world network rewired from the Moore lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import LatticeElement
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import both, either, neighbors, where
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec
from ....core.types import AnyArray
from ..._catalog import register_system
from ._binary_cell import BinaryCellElement

MOORE_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))


BIRTH, SURVIVAL = 3, 2


def _moore_torus(shape: Sequence[int]) -> nx.Graph:
    """Return the graph of the cells of a torus, each linked to its eight Moore neighbours, in row-major order."""
    rows, columns = shape
    graph = nx.Graph()
    graph.add_nodes_from(range(rows * columns))
    for row in range(rows):
        for column in range(columns):
            for down, right in MOORE_OFFSETS:
                graph.add_edge(row * columns + column, (row + down) % rows * columns + (column + right) % columns)
    return graph


def _life(s: SimpleNamespace) -> Expression:
    """Return the next state of a cell of the Game of Life from its live neighbours on the network."""
    live = neighbors(s.target.other.alive)
    born_or_kept = either(live == BIRTH, both(s.target.alive == 1, live == SURVIVAL))
    alive: Expression = where(born_or_kept, 1, 0)
    return alive


def _alive_fraction(alive: AnyArray) -> float:
    """Return the fraction of live cells."""
    return float(np.mean(np.asarray(alive, dtype=float)))


@register_system(
    field='computer_science/automata',
    tags=('network', 'life', 'small world', 'emergence'),
    references=('Watts and Strogatz 1998', 'Gros 2015'),
    defaults={'until': 100, 'dt': 1},
    observables=lambda system: [
        ObservableSpec('density', group='cells', variable='alive', reduce=_alive_fraction),
    ],
)
def small_world_life(
    shape: Sequence[int] = (50, 50),
    *,
    p: float = 0.0,
    density: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Conway's Game of Life on a small-world network rewired from the Moore lattice of a torus.

    Every cell is a node linked to its eight Moore neighbours. The rewiring cuts two random links and joins the four
    loose ends the other way, which keeps every node at eight neighbours, and repeats it `p` times half the number of
    links, so a fraction of about `p` of the links becomes shortcuts across the torus. The rules of life are the same
    on the network: a dead cell with three live neighbours comes to life, and a live cell with two or three survives.
    For `p = 0` the network is the lattice and the automaton is the Game of Life, and the shortcuts change the density
    of live cells it settles at, which serves as an order parameter of the rewiring.

    Args:
        shape:
            The number of rows and columns of the torus.

        p:
            The rewiring probability, the fraction of the links rewired.

        density:
            The probability that a cell starts alive.

        random_state:
            The seed or generator of the rewiring and of the initial cells.

    Returns:
        system:
            The system, with the cells `cells.alive`, one per node in row-major order of the torus. Its standard
            observable `density` is the fraction of live cells.

    Raises:
        ParameterError:
            The rewiring probability is outside `[0, 1]`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import small_world_life
        >>> sorted({degree for _, degree in small_world_life((10, 10), p=0.2).elements[0].space.graph.degree()})
        [8]
    """
    if not 0 <= p <= 1:
        error_msg = f'The rewiring probability is {p!r}. Give a number in [0, 1].'
        raise ParameterError(error_msg)
    rng = np.random.default_rng(random_state)
    graph = _moore_torus(shape)
    swaps = round(p * graph.number_of_edges() / 2)
    if swaps:
        nx.double_edge_swap(graph, nswap=swaps, max_tries=100 * swaps, seed=int(rng.integers(2**31)))
    alive = (rng.random(graph.number_of_nodes()) < density).astype(float)
    cells = LatticeElement('cells', element=BinaryCellElement, space=NetworkSpace(graph), values={'alive': alive})
    return System(
        elements=[cells],
        interactions=[MapInteraction('life', target='cells', state='alive', function=_life)],
        time=DiscreteTime(),
    )
