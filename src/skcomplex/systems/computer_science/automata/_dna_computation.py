"""DNA computation: Adleman's test tube, which solves a Hamiltonian path.

Random ligation of edge oligos into paths solves the problem.

`ADLEMAN_EDGES` holds the directed links of the seven-node graph of Adleman's experiment, whose only Hamiltonian path is
0 to 6 in order.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import AnyArray, IntArray

ADLEMAN_EDGES = (
    (0, 1),
    (0, 3),
    (0, 6),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 1),
    (4, 5),
    (5, 1),
    (5, 2),
    (5, 6),
)
EMPTY = -1


class StrandView(AgentView):
    """View of a strand of DNA, with its path as an attribute."""

    path: IntArray


class StrandElement(Element):
    """Strand of DNA in the test tube.

    It holds the path of nodes its ligated edge oligos spell, padded with -1.

    Args:
        label:
            The label of the strand or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        path:
            The initial nodes of the path, padded with -1 to the longest path the tube holds.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import StrandElement
        >>> StrandElement('strand', path=(0, 1, -1)).path
        (0, 1, -1)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        path: Annotated[object, StateSpec()] = EMPTY,
    ) -> None:
        super().__init__(label, visual=visual)
        self.path = path

    @classmethod
    def _get_view(cls) -> type[StrandView]:
        """Return the view of the strands.

        It types their path.
        """
        return StrandView


class LigationInteraction(Interaction):
    """Random ligation of the strands of a test tube.

    It joins a strand that ends at a node to one that starts there. Every step the strands meet in random pairs. The
    oligo of the node that the first strand ends at and the second starts at holds their ends together, and the ligase
    joins them into one strand, as long as the joined path fits the longest path the tube holds. The second strand is
    consumed, and an edge oligo drawn uniformly from the links of the graph takes its place, since the tube holds a vast
    excess of oligos.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of strands.

        edges:
            The directed links of the graph, one edge oligo each.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LigationInteraction
        >>> LigationInteraction('ligation', target='strands', edges=((0, 1),)).edges
        ((0, 1),)
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        edges: Sequence[tuple[int, int]] = ADLEMAN_EDGES,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.edges = edges

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the ligations of a step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rule of the ligations.
        """
        return [RuleLaw(self._ligate)]

    def _ligate(self, system: SystemView) -> None:
        """Join the strands of random pairs whose ends meet.

        Every consumed strand is replaced with a fresh oligo.
        """
        strands = system.get_agents('target', StrandView)
        rng = system.rng
        order = rng.permutation(len(strands))
        for first, second in zip(order[::2], order[1::2], strict=False):
            head, tail = np.asarray(strands[first].path), np.asarray(strands[second].path)
            head_length, tail_length = int((head >= 0).sum()), int((tail >= 0).sum())
            if head[head_length - 1] != tail[0] or head_length + tail_length - 1 > head.size:
                continue
            joined = head.copy()
            joined[head_length : head_length + tail_length - 1] = tail[1:tail_length]
            strands[first].path = joined
            strands[second].path = self._get_oligo(head.size, rng)

    def _get_oligo(self, size: int, rng: np.random.Generator) -> IntArray:
        """Return a fresh edge oligo drawn uniformly from the links.

        It is a path of two nodes padded to the size.
        """
        path = np.full(size, EMPTY)
        path[:2] = self.edges[int(rng.integers(len(self.edges)))]
        return path


def _filtered_counts(paths: AnyArray, start: int, end: int, n_nodes: int) -> list[int]:
    """Return the numbers of strands left after each of Adleman's filters.

    The filters are by start and end, length, and coverage.
    """
    nodes: IntArray = np.asarray(paths, dtype=np.int64)
    lengths = (nodes >= 0).sum(axis=1)
    rows = np.arange(len(nodes))
    ends = (nodes[:, 0] == start) & (nodes[rows, np.maximum(lengths - 1, 0)] == end)
    sized = ends & (lengths == n_nodes)
    return [
        int(ends.sum()),
        int(sized.sum()),
        int(_get_solutions(nodes, start, end, n_nodes).sum()),
    ]


def _get_adleman_solutions(paths: AnyArray) -> float:
    """Return the number of strands that are the Hamiltonian path of Adleman's graph.

    The path goes from node 0 to node 6.
    """
    return float(_get_solutions(paths, 0, 6, 7).sum())


def _get_solutions(paths: AnyArray, start: int, end: int, n_nodes: int) -> IntArray:
    """Return the mask of the strands that pass Adleman's filters.

    They are the Hamiltonian paths from the start to the end.
    """
    nodes: IntArray = np.asarray(paths, dtype=np.int64)
    lengths = (nodes >= 0).sum(axis=1)
    rows = np.arange(len(nodes))
    starting = nodes[:, 0] == start
    ending = nodes[rows, np.maximum(lengths - 1, 0)] == end
    sized = lengths == n_nodes
    covered = np.array([np.unique(path[path >= 0]).size == n_nodes for path in nodes])
    return starting & ending & sized & covered


class DNAComputationSystem(BaseSystem):
    """Adleman's DNA computation of a Hamiltonian path.

    It runs by random ligation of edge oligos in a test tube. Every node of a directed graph is an oligonucleotide, and
    every link an oligo that joins the end of one node to the start of the next, so ligation builds strands that spell
    random paths of the graph. The tube starts with edge oligos drawn uniformly from the links, which meet in random
    pairs and join when the end of one is the start of the other. Adleman's filters then keep the strands that start at
    `start` and end at `end`, that pass through as many nodes as the graph has, and that visit every node, and any
    strand left is a Hamiltonian path. On the graph of Adleman's experiment, with seven nodes and fourteen links,
    solutions appear after some steps and every one of them is the path 0, 1, 2, 3, 4, 5, 6, while a graph without a
    Hamiltonian path leaves no strand. The system has the group `strands` of paths `path`, padded with -1. Its standard
    observable `solutions` is the number of strands that pass the filters for the default nodes of Adleman's graph.

    Args:
        n_strands:
            The number of strands in the tube.

        edges:
            The directed links of the graph, with the nodes numbered from 0.

        start:
            The node the paths start at.

        end:
            The node the paths end at.

        random_state:
            The seed or generator of the initial oligos.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ParameterError:
            The graph has no links, or the start or the end is not a node of it.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import DNAComputationSystem
        >>> DNAComputationSystem(10).prepare().system_.elements[0].values['path'].shape
        (10, 7)
    """

    field = 'computer_science/automata'
    topics = ('dna computing', 'molecular computation', 'hamiltonian path', 'agents', 'stochastic')
    references = ('Adleman 1994', 'Raman 2021')
    until = 60
    observables = (ObservableSpec('solutions', group='strands', variable='path', reduce=_get_adleman_solutions),)

    def __init__(
        self,
        n_strands: int = 2000,
        *,
        edges: Sequence[tuple[int, int]] = ADLEMAN_EDGES,
        start: int = 0,
        end: int = 6,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_strands = n_strands
        self.edges = edges
        self.start = start
        self.end = end
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return Adleman's DNA computation of a Hamiltonian path.

        It runs by random ligation of edge oligos in a test tube.
        """
        nodes = {node for edge in self.edges for node in edge}
        if not self.edges or self.start not in nodes or self.end not in nodes:
            error_msg = f'The start {self.start!r} and the end {self.end!r} need to be nodes of a graph with links.'
            raise ParameterError(error_msg)
        rng = np.random.default_rng(self.random_state)
        paths = np.full((self.n_strands, len(nodes)), EMPTY)
        paths[:, :2] = np.asarray(self.edges)[rng.integers(len(self.edges), size=self.n_strands)]
        strands = GroupElement('strands', element=StrandElement, size=self.n_strands, values={'path': paths})
        return CompositeSystem(
            self.label,
            elements=[strands],
            interactions=[LigationInteraction('ligation', target='strands', edges=self.edges)],
        )
