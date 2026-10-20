"""Declare rewriting systems: words, patches of tiles, and host graphs rewritten in parallel by productions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..base import RewritingError, VisualSpec, role
from ._components import Element, Interaction
from ._symbols import Symbols

ACTIONS = frozenset(
    {
        'draw',
        'move',
        'left',
        'right',
        'reverse',
        'push',
        'pop',
        'pitch_down',
        'pitch_up',
        'roll_left',
        'roll_right',
    },
)
TURTLE_SYMBOLS: dict[str, str] = {
    'F': 'draw',
    'f': 'move',
    '+': 'left',
    '-': 'right',
    '|': 'reverse',
    '[': 'push',
    ']': 'pop',
    '&': 'pitch_down',
    '^': 'pitch_up',
    '\\': 'roll_left',
    '/': 'roll_right',
}


class Module:
    """Symbol of a word with numeric or symbolic parameters, such as `A(x)` or `F(1.0)`.

    Args:
        name:
            The name of the symbol.

        *parameters:
            The parameters, numbers in a word and symbols or expressions in a production.

    Examples:
        >>> from skcomplex.core.model import Module
        >>> Module('F', 0.5)
        F(0.5)
    """

    def __init__(self, name: str, *parameters: Any) -> None:  # noqa: ANN401
        self.name = name
        self.parameters = parameters

    def __repr__(self) -> str:
        """Return the module as written, such as `A(x, 2)`."""
        if not self.parameters:
            return self.name
        return f'{self.name}({", ".join(str(item) for item in self.parameters)})'

    def __eq__(self, other: object) -> bool:
        """Check that two modules have the same name and parameters."""
        return isinstance(other, Module) and (self.name, self.parameters) == (other.name, other.parameters)

    def __hash__(self) -> int:
        """Return the hash of the name."""
        return hash(self.name)


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
        >>> from skcomplex.core.model import Module
        >>> from skcomplex.core.model._rewriting import parse_word
        >>> parse_word(['F', Module('A', 1.0), '+-'])
        [F, A(1.0), +, -]
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
class Production:
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
        >>> from skcomplex.core.model import Production
        >>> Production('F', 'F+F--F+F')
        F → F+F--F+F
    """

    predecessor: str | Module
    successor: Letters
    left: Letters | None = None
    right: Letters | None = None
    condition: Any = None
    probability: Any = None

    def __repr__(self) -> str:
        """Return the production as written, such as `B < A → B`."""
        head = repr(self.predecessor if isinstance(self.predecessor, Module) else Module(self.predecessor))
        if self.left is not None:
            head = f'{"".join(map(repr, parse_word(self.left)))} < {head}'
        if self.right is not None:
            head = f'{head} > {"".join(map(repr, parse_word(self.right)))}'
        modules = parse_word(self.successor)
        separator = ' ' if any(module.parameters for module in modules) else ''
        text = f'{head} → {separator.join(map(repr, modules))}'
        if self.condition is not None:
            text += f' : {self.condition}'
        if self.probability is not None:
            text += f' ({self.probability})'
        return text


@dataclass(frozen=True)
class Turtle:
    r"""Turtle that interprets a word as line segments in two or three dimensions.

    Symbols mean `draw` (forward with a segment), `move` (forward without one), `left` and `right` (turn by the angle),
    `reverse` (turn by 180 degrees), `push` and `pop` (save and restore the position and heading), and in three
    dimensions `pitch_down`, `pitch_up`, `roll_left`, and `roll_right`. The default meanings are those of
    Prusinkiewicz and Lindenmayer (1990): `F`, `f`, `+`, `-`, `|`, `[`, `]`, `&`, `^`, `\`, and `/`. A module's first
    parameter overrides the step of a move or the angle of a turn.

    Args:
        angle:
            The turning angle in degrees.

        step:
            The step at generation 0.

        contraction:
            The factor the step is multiplied by at every generation.

        symbols:
            Meanings of symbols that replace or extend the default ones.

        dimension:
            2 or 3.

        heading:
            The initial heading in degrees in two dimensions, measured from the x axis.

    Examples:
        >>> from skcomplex.core.model import Turtle
        >>> Turtle(angle=60.0, contraction=1 / 3).meanings()['F']
        'draw'
    """

    angle: float = 90.0
    step: float = 1.0
    contraction: float = 1.0
    symbols: Mapping[str, str] | None = None
    dimension: int = 2
    heading: float = 0.0

    def meanings(self) -> dict[str, str]:
        """Return the meaning of every symbol the turtle interprets.

        Returns:
            meanings:
                A mapping from symbol to action.

        Raises:
            RewritingError:
                A meaning is not an action of the turtle, or the dimension is neither 2 nor 3.
        """
        meanings = {**TURTLE_SYMBOLS, **(self.symbols or {})}
        unknown = sorted(set(meanings.values()) - ACTIONS)
        if unknown:
            error_msg = f'The turtle has no action {unknown[0]!r}. Use one of {", ".join(sorted(ACTIONS))}.'
            raise RewritingError(error_msg)
        if self.dimension not in {2, 3}:
            error_msg = f'The turtle has dimension {self.dimension}. Use 2 or 3.'
            raise RewritingError(error_msg)
        return meanings


@dataclass(frozen=True)
class Prototile:
    """Shape that tiles of a patch are placed copies of, a segment or a polygon in its own frame.

    Args:
        name:
            The name of the prototile, which names its count state.

        vertices:
            The vertices in the plane, two for a segment and at least three for a polygon.

    Examples:
        >>> from skcomplex.core.model import Prototile
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
        >>> from skcomplex.core.model import Affine
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
class Substitution:
    """Substitution that replaces every tile of a prototile by pieces placed relative to the tile.

    Args:
        prototile:
            The name of the prototile replaced.

        pieces:
            Pairs of a prototile name and the affine map from the parent tile's frame.

    Examples:
        >>> from skcomplex.core.model import Affine, Substitution
        >>> len(Substitution('interval', [('interval', Affine(scale=1 / 3))]).pieces)
        1
    """

    prototile: str
    pieces: Sequence[tuple[str, Affine]]

    def __repr__(self) -> str:
        """Return the substitution as written."""
        return f'{self.prototile} → {", ".join(name for name, _ in self.pieces)}'


@dataclass(frozen=True, eq=False)
class GraphProduction:
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
        >>> from skcomplex.core.model import GraphProduction
        >>> left = nx.Graph([('a', 'b')])
        >>> interface = nx.Graph()
        >>> interface.add_nodes_from('ab')
        >>> production = GraphProduction(left, interface, nx.Graph([('a', 'c'), ('c', 'b')]))
        >>> sorted(production.added())
        ['c']
    """

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

    def __repr__(self) -> str:
        """Return the sizes of the three graphs."""
        sizes = ' ← '.join(f'{graph.number_of_nodes()} nodes' for graph in (self.left, self.interface))
        return f'GraphProduction({sizes} → {self.right.number_of_nodes()} nodes)'


class Word(Element):
    """Element that holds a word of modules rewritten in parallel by string productions.

    Its states are the count of every symbol whose name is a word of letters, digits, and underscores, and the length
    `size` of the word.

    Args:
        label:
            The label of the element.

        axiom:
            The word at generation 0.

        alphabet:
            The symbols, as a string of one-character symbols or a mapping from name to number of parameters, or
            `None` to take them from the axiom and the productions.

        ignore:
            The symbols that contexts skip.

        turtle:
            The turtle that interprets the word as geometry, or `None`.

        max_size:
            The largest word a generation may produce.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` the geometry visual shows, or `None` to fit the data.

    Examples:
        >>> from skcomplex.core.model import Word
        >>> Word('algae', axiom='A').axiom
        'A'
    """

    visual: ClassVar[VisualSpec] = VisualSpec('geometry')
    axiom: Letters = ''
    alphabet: str | Mapping[str, int] | None = None
    ignore: str | Sequence[str] = ''
    turtle: Turtle | None = None
    max_size: int = 10_000_000
    bounds: Sequence[float] | None = None


class Patch(Element):
    """Element that holds tiles, each a prototile under an affine placement, replaced in parallel by substitutions.

    Its states are the count of every prototile and the number `size` of tiles.

    Args:
        label:
            The label of the element.

        prototiles:
            The prototiles.

        tiles:
            Pairs of a prototile name and its placement at generation 0, or `None` for the first prototile in place.

        filled:
            Whether the geometry visual draws polygons as a filled image instead of outlines.

        max_size:
            The largest number of tiles a generation may produce.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` the geometry visual shows, or `None` for the box of the initial tiles.

    Examples:
        >>> from skcomplex.core.model import Patch, Prototile
        >>> Patch('cantor', prototiles=[Prototile('interval', [(0, 0), (1, 0)])]).filled
        False
    """

    visual: ClassVar[VisualSpec] = VisualSpec('geometry')
    prototiles: Sequence[Prototile] = ()
    tiles: Sequence[tuple[str, Affine]] | None = None
    filled: bool = False
    max_size: int = 10_000_000
    bounds: Sequence[float] | None = None


class HostGraph(Element):
    """Element that holds a graph rewritten by double-pushout graph productions.

    Its states are the count of nodes of every `label` attribute that is a word, and the number `size` of nodes.

    Args:
        label:
            The label of the element.

        graph:
            The graph at generation 0, directed or not.

        max_size:
            The largest number of nodes a generation may produce.

        max_matches:
            The largest number of matches searched per production and generation.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.model import HostGraph
        >>> HostGraph('ring', graph=nx.cycle_graph(4)).graph.number_of_nodes()
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('geometry')
    graph: nx.Graph = field(default_factory=nx.Graph)
    max_size: int = 1_000_000
    max_matches: int = 100_000


type RewritingLaw = Production | Substitution | GraphProduction


class Rewriting(Interaction):
    """Base class of interactions that rewrite the word, patch, or host graph bound to their `target` role.

    Subclasses declare parameters with `parameter` and override `productions`, where `s.<parameter>` is a symbol that
    conditions, successor parameters, probabilities, and affine maps can use.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the rewritten element.

    Examples:
        >>> from skcomplex.core.base import parameter
        >>> from skcomplex.core.model import Module, Production, Rewriting
        >>> from skcomplex.core.base import symbol
        >>> class Halving(Rewriting):
        ...     ratio: float = parameter(0.5)
        ...
        ...     def productions(self, s):
        ...         x = symbol('x')
        ...         return [Production(Module('A', x), [Module('F', x), Module('A', s.ratio * x)])]
        >>> Halving('halving', target='plant').ratio
        0.5
    """

    target: str = role()

    def productions(self, s: Symbols) -> list[RewritingLaw]:
        """Return the productions or substitutions of the interaction.

        Args:
            s:
                The namespace of the parameter symbols.

        Returns:
            productions:
                The productions, substitutions, or graph productions.
        """
        del s
        return []


class Rewrite(Rewriting):
    """Interaction that rewrites its target by fixed productions, substitutions, or graph productions.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the rewritten element.

        rules:
            The productions, substitutions, or graph productions.

    Examples:
        >>> from skcomplex.core.model import Production, Rewrite
        >>> Rewrite('rules', target='algae', rules=[Production('A', 'AB'), Production('B', 'A')]).rules[1]
        B → A
    """

    rules: Sequence[RewritingLaw] = ()

    def productions(self, s: Symbols) -> list[RewritingLaw]:
        """Return the fixed rules.

        Args:
            s:
                The namespace of the parameter symbols.

        Returns:
            productions:
                The rules.
        """
        del s
        return list(self.rules)
