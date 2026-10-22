"""Define the elements that rewriting systems rewrite: words, patches of tiles, and host graphs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar

import networkx as nx

from ..base import (
    Structure,
    VariableKind,
    VisualSpec,
    declarations_of,
    state,
)
from ..laws._rewriting import Affine, Letters, Prototile
from ._components import Element

if TYPE_CHECKING:
    from ..viz import WordGeometry


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

        geometry:
            The interpretation of the word as line segments, or `None`.

        max_size:
            The largest word a generation may produce.

        bounds:
            The box `(xmin, ymin, xmax, ymax)` the geometry visual shows, or `None` to fit the data.

    Examples:
        >>> from skcomplex.core.components import Word
        >>> Word('algae', axiom='A').axiom
        'A'
    """

    visual: ClassVar[VisualSpec] = VisualSpec('geometry')
    axiom: Letters = state('', domain=Structure('sequence'))
    alphabet: str | Mapping[str, int] | None = None
    ignore: str | Sequence[str] = ''
    geometry: 'WordGeometry | None' = None
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
        >>> from skcomplex.core.components import Patch
        >>> from skcomplex.core.laws import Prototile
        >>> Patch('cantor', prototiles=[Prototile('interval', [(0, 0), (1, 0)])]).filled
        False
    """

    visual: ClassVar[VisualSpec] = VisualSpec('geometry')
    prototiles: Sequence[Prototile] = ()
    tiles: Sequence[tuple[str, Affine]] | None = state(default=None, domain=Structure('tiling'))
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
        >>> from skcomplex.core.components import HostGraph
        >>> HostGraph('ring', graph=nx.cycle_graph(4)).graph.number_of_nodes()
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('geometry')
    graph: nx.Graph = state(nx.Graph(), domain=Structure('graph'))
    max_size: int = 1_000_000
    max_matches: int = 100_000


def structure_of(element_type: type) -> str | None:
    """Return the kind of structure an element type rewrites, from the domain of its structured state.

    Args:
        element_type:
            The element type.

    Returns:
        kind:
            `'sequence'`, `'tiling'`, or `'graph'`, or `None` for a type without a structured state.

    Examples:
        >>> from skcomplex.core.components import Word, structure_of
        >>> structure_of(Word)
        'sequence'
    """
    if not (isinstance(element_type, type) and issubclass(element_type, Element)):
        return None
    for declaration in declarations_of(element_type, VariableKind.STATE).values():
        if isinstance(declaration.domain, Structure):
            return declaration.domain.kind
    return None
