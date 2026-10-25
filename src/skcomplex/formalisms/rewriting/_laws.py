"""Define rewriting laws: productions on words, substitutions of tiles, and graph productions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ...core.base import BaseLaw, DomainKind, LawClass, LawKind, Requires, SkcomplexError, TimeKind


class RewritingError(SkcomplexError):
    """A rewriting system is malformed, or a generation would exceed its maximum size."""


@dataclass(frozen=True)
class Module:
    """Symbol of a word with numeric or symbolic parameters, such as `A(x)` or `F(1.0)`.

    Args:
        name:
            The name of the symbol.

        parameters:
            The parameters, numbers in a word and symbols or expressions in a production.

    Examples:
        >>> from skcomplex.formalisms.rewriting import Module
        >>> Module('F', (0.5,)).text
        'F(0.5)'
    """

    name: str
    parameters: tuple[Any, ...] = ()

    @property
    def text(self) -> str:
        """Module as written, such as `A(x, 2)`."""
        if not self.parameters:
            return self.name
        return f'{self.name}({", ".join(str(item) for item in self.parameters)})'


type Letters = str | Module | Sequence[str | Module]


def parse_word(value: Letters, names: Sequence[str] = ()) -> list[Module]:
    """Return the modules of a word given as a string of one-character symbols, a module, or a sequence of both.

    A string item of a sequence is one symbol when it is a declared name or one character long, and a string of
    one-character symbols otherwise.

    Args:
        value:
            The word.

        names:
            The declared symbol names, which may be longer than one character.

    Returns:
        modules:
            The modules of the word.

    Examples:
        >>> from skcomplex.formalisms.rewriting import Module
        >>> from skcomplex.formalisms.rewriting._laws import parse_word
        >>> [module.text for module in parse_word(['F', Module('A', (1.0,)), '+-'])]
        ['F', 'A(1.0)', '+', '-']
    """
    if isinstance(value, Module):
        return [value]
    if isinstance(value, str):
        return [Module(value)] if value in names else [Module(item) for item in value]
    modules: list[Module] = []
    for item in value:
        modules.extend(parse_word(item, names))
    return modules


@dataclass(frozen=True, eq=False)
class ProductionLaw(BaseLaw):
    """Production of a string rewriting system, which replaces every matching module of a word by a successor.

    The predecessor's parameters are symbols bound to the parameters of each matching module. The condition, the
    successor's parameters, and the probability are expressions of these symbols and of the parameters of the
    interaction that returns the production. A production with a left or a right context matches only modules whose
    neighbours, skipping the ignored symbols of the word and its bracketed branches, spell the context.

    Args:
        predecessor:
            The symbol, or a module whose parameters are symbols.

        successor:
            The word that replaces the predecessor.

        left:
            The left context, or `None`.

        right:
            The right context, or `None`.

        condition:
            A symbolic condition of the predecessor's parameters, or `None`.

        probability:
            The probability of a stochastic production, or `None` for a deterministic one. The probabilities of the
            stochastic productions of one predecessor, context, and condition sum to one.

    Examples:
        >>> from skcomplex.formalisms.rewriting import ProductionLaw
        >>> ProductionLaw('F', 'F+F--F+F').text
        'F → F+F--F+F'
    """

    law_kind: ClassVar[LawKind] = LawKind.REWRITE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, domains={DomainKind.SEQUENCE})

    predecessor: str | Module
    successor: Letters
    left: Letters | None = None
    right: Letters | None = None
    condition: Any = None
    probability: Any = None

    @property
    def text(self) -> str:
        """Production as written, such as `B < A → B`."""
        head = (self.predecessor if isinstance(self.predecessor, Module) else Module(self.predecessor)).text
        if self.left is not None:
            head = f'{"".join(module.text for module in parse_word(self.left))} < {head}'
        if self.right is not None:
            head = f'{head} > {"".join(module.text for module in parse_word(self.right))}'
        modules = parse_word(self.successor)
        separator = ' ' if any(module.parameters for module in modules) else ''
        text = f'{head} → {separator.join(module.text for module in modules)}'
        if self.condition is not None:
            text += f' : {self.condition}'
        if self.probability is not None:
            text += f' ({self.probability})'
        return text


@dataclass(frozen=True)
class Prototile:
    """Shape that tiles of a patch are placed copies of, a segment or a polygon in its own frame.

    Args:
        name:
            The name of the prototile, which names its count state.

        vertices:
            The vertices in the plane, two for a segment and at least three for a polygon.

    Examples:
        >>> from skcomplex.formalisms.rewriting import Prototile
        >>> Prototile('interval', [(0, 0), (1, 0)]).is_segment
        True
    """

    name: str
    vertices: Sequence[Sequence[float]]

    @property
    def is_segment(self) -> bool:
        """Whether the prototile is a segment."""
        return len(self.vertices) == 2  # noqa: PLR2004

    def points(self) -> npt.NDArray[np.float64]:
        """Return the vertices as an array.

        Returns:
            points:
                The array of shape `(k, 2)`.

        Raises:
            RewritingError:
                The vertices are not points of the plane, or fewer than two.
        """
        points = np.asarray(self.vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:  # noqa: PLR2004
            error_msg = f'Prototile {self.name!r} needs at least two vertices in the plane, but got {self.vertices!r}.'
            raise RewritingError(error_msg)
        return points


@dataclass(frozen=True)
class Affine:
    """Affine map of the plane: a reflection across the x axis, a rotation, a scale, and a translation, in order.

    The scale, the rotation, and the translation can be expressions of the parameters of the interaction that
    returns the substitution, which are evaluated at every generation.

    Args:
        scale:
            The scale factor.

        rotation:
            The rotation in degrees.

        translation:
            The translation.

        reflect:
            Whether to reflect across the x axis first.

        matrix:
            A homogeneous 3 by 3 matrix that replaces the other arguments.

    Examples:
        >>> from skcomplex.formalisms.rewriting import Affine
        >>> Affine(scale=0.5, translation=(1, 0)).matrix_of({}).tolist()
        [[0.5, 0.0, 1.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
    """

    scale: Any = 1.0
    rotation: Any = 0.0
    translation: Sequence[Any] = (0.0, 0.0)
    reflect: bool = False
    matrix: Sequence[Sequence[float]] | None = None

    def matrix_of(self, values: Mapping[Any, float]) -> npt.NDArray[np.float64]:
        """Return the homogeneous matrix of the map for values of the parameters.

        Args:
            values:
                A mapping from parameter symbol to value.

        Returns:
            matrix:
                The 3 by 3 matrix.
        """
        if self.matrix is not None:
            return np.asarray(self.matrix, dtype=float)

        def number(item: Any) -> float:  # noqa: ANN401
            return float(item.subs(values)) if hasattr(item, 'subs') else float(item)

        scale, angle = number(self.scale), np.deg2rad(number(self.rotation))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        reflection = np.diag([1.0, -1.0]) if self.reflect else np.eye(2)
        matrix = np.eye(3)
        matrix[:2, :2] = scale * rotation @ reflection
        matrix[:2, 2] = [number(item) for item in self.translation]
        return matrix


@dataclass(frozen=True, eq=False)
class SubstitutionLaw(BaseLaw):
    """Substitution that replaces every tile of a prototile by pieces placed relative to the tile.

    Args:
        prototile:
            The name of the prototile replaced.

        pieces:
            Pairs of a prototile name and the affine map from the parent tile's frame.

    Examples:
        >>> from skcomplex.formalisms.rewriting import Affine, SubstitutionLaw
        >>> len(SubstitutionLaw('interval', [('interval', Affine(scale=1 / 3))]).pieces)
        1
    """

    law_kind: ClassVar[LawKind] = LawKind.REWRITE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, domains={DomainKind.TILING})

    prototile: str
    pieces: Sequence[tuple[str, Affine]]

    @property
    def text(self) -> str:
        """Substitution as written, such as `interval → interval, interval`."""
        return f'{self.prototile} → {", ".join(name for name, _ in self.pieces)}'


@dataclass(frozen=True, eq=False)
class GraphProductionLaw(BaseLaw):
    """Double-pushout graph production: a left graph, an interface, and a right graph that share nodes by name.

    A match of the left graph in a host graph deletes the nodes and edges of the left graph that are not in the
    interface and adds the nodes and edges of the right graph that are not in it. Nodes and edges match when their
    `label` attributes agree, where the left graph gives one.

    Args:
        left:
            The left graph.

        interface:
            The interface, whose nodes are nodes of both other graphs.

        right:
            The right graph.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.formalisms.rewriting import GraphProductionLaw
        >>> left = nx.Graph([('a', 'b')])
        >>> interface = nx.Graph()
        >>> interface.add_nodes_from('ab')
        >>> production = GraphProductionLaw(left, interface, nx.Graph([('a', 'c'), ('c', 'b')]))
        >>> sorted(production.added())
        ['c']
    """

    law_kind: ClassVar[LawKind] = LawKind.REWRITE
    time_class: ClassVar[LawClass] = LawClass.DISCRETE
    requires: ClassVar[Requires] = Requires(time={TimeKind.DISCRETE}, domains={DomainKind.GRAPH})

    left: nx.Graph
    interface: nx.Graph
    right: nx.Graph

    def added(self) -> set[Any]:
        """Return the nodes the production adds.

        Returns:
            nodes:
                The nodes of the right graph outside the interface.
        """
        return set(self.right.nodes) - set(self.interface.nodes)


type RewritingLaw = ProductionLaw | SubstitutionLaw | GraphProductionLaw
