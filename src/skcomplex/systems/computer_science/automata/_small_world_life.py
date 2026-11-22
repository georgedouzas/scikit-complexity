"""Automata systems on networks: the Game of Life on a small-world network.

The network is rewired from the Moore lattice.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import networkx as nx
import numpy as np

from ....core.components import Observable
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import both, either, neighbors, where
from ....core.components.spaces import NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ParameterError
from ....core.systems import BaseSystem, System
from ....core.types import AnyArray, Role
from ._binary_cell import BinaryCellElement

MOORE_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))
BIRTH, SURVIVAL = 3, 2


def _get_alive_fraction(alive: AnyArray) -> float:
    """Return the fraction of live cells."""
    return float(np.mean(np.asarray(alive, dtype=float)))


class NetworkLifeInteraction(Interaction):
    """Rules of Conway's Game of Life on the nodes of a network.

    A dead cell with three live neighbours comes to life, a live cell with two or three live neighbours survives, and
    every other cell is dead at the next step. The neighbours are those of the node on the network.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `alive` the rules update.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import NetworkLifeInteraction
        >>> NetworkLifeInteraction('life', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next state of every cell, read from its live neighbours, as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        alive = context.target.alive
        live = neighbors(context.target.other.alive)
        born_or_kept = either(live == BIRTH, both(alive == 1, live == SURVIVAL))
        self.laws_: list[BaseLaw] = [UpdateLaw(alive, where(born_or_kept, 1, 0))]


class SmallWorldLifeSystem(BaseSystem):
    """Conway's Game of Life on a small-world network.

    The network is rewired from the Moore lattice of a torus.

    Every cell is a node linked to its eight Moore neighbours. The rewiring cuts two random links and joins the four
    loose ends the other way, which keeps every node at eight neighbours, and repeats it `p` times half the number of
    links, so a fraction of about `p` of the links becomes shortcuts across the torus. The rules of life are the same on
    the network: a dead cell with three live neighbours comes to life, and a live cell with two or three survives. For
    `p = 0` the network is the lattice and the automaton is the Game of Life, and the shortcuts change the density of
    live cells it settles at, which serves as an order parameter of the rewiring. The system has the cells
    `cells.alive`, one per node in row-major order of the torus. Its standard observable `density` is the fraction of
    live cells.

    Args:
        shape:
            The number of rows and columns of the torus.

        p:
            The rewiring probability, the fraction of the links rewired.

        density:
            The probability that a cell starts alive.

        random_state:
            The seed or generator of the rewiring and of the initial cells.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The rewiring probability is outside `[0, 1]`.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import SmallWorldLifeSystem
        >>> graph = SmallWorldLifeSystem((10, 10), p=0.2).prepare().system_.elements[0].space.graph
        >>> sorted({degree for _, degree in graph.degree()})
        [8]
    """

    field = 'computer_science/automata'
    topics = ('network', 'life', 'small world', 'emergence')
    references = ('Watts and Strogatz 1998', 'Gros 2015')
    until = 100
    dt = 1
    observables = (Observable('density', group='cells', variable='alive', reduce=_get_alive_fraction),)

    def __init__(
        self,
        shape: Sequence[int] = (50, 50),
        *,
        p: float = 0.0,
        density: float = 0.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.p = p
        self.density = density
        self.random_state = random_state

    @staticmethod
    def _get_moore_torus(shape: Sequence[int]) -> nx.Graph:
        """Return the graph of the cells of a torus.

        Each cell is linked to its eight Moore neighbours, in row-major order.
        """
        rows, columns = shape
        graph = nx.Graph()
        graph.add_nodes_from(range(rows * columns))
        for row in range(rows):
            for column in range(columns):
                for down, right in MOORE_OFFSETS:
                    graph.add_edge(row * columns + column, (row + down) % rows * columns + (column + right) % columns)
        return graph

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells on a rewired Moore torus and the rules of life."""
        super()._prepare(context)
        if not 0 <= self.p <= 1:
            error_msg = f'The rewiring probability is {self.p!r}. Give a number in [0, 1].'
            raise ParameterError(error_msg)
        rng = np.random.default_rng(self.random_state)
        graph = self._get_moore_torus(self.shape)
        swaps = round(self.p * graph.number_of_edges() / 2)
        if swaps:
            nx.double_edge_swap(graph, nswap=swaps, max_tries=100 * swaps, seed=int(rng.integers(2**31)))
        alive = (rng.random(graph.number_of_nodes()) < self.density).astype(float)
        self.system_ = System(
            self.label,
            elements=[BinaryCellElement('cells', alive=alive, space=NetworkSpace(graph))],
            interactions=[NetworkLifeInteraction('life', target='cells')],
        )
