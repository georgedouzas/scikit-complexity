"""Compile rewriting systems: words, patches, and host graphs with their productions, as problems of generations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from typing import Any, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.algorithms import isomorphism

from ..base import (
    Domain,
    Form,
    IncompatibleLawsError,
    RewritingError,
    RoleError,
    ValidationError,
    VariableKind,
    declarations_of,
    symbol,
)
from ..model import (
    Element,
    GraphProduction,
    Group,
    HostGraph,
    Patch,
    Production,
    Rewriting,
    RewritingLaw,
    Substitution,
    Symbols,
    System,
    Word,
)
from ._layout import Entry, Layout
from ._problem import CompiledElement, CompiledInteraction, Problem
from ._structures import GraphStructure, PatchStructure
from ._tree import Node, flatten_system
from ._words import IDENTIFIER, CompiledWord, Parameters

REWRITTEN = (Word, Patch, HostGraph)


class CompiledPatch:
    """Substitution system of one patch, compiled to batched products of affine matrices.

    Args:
        path:
            The path of the patch.

        patch:
            The patch element.

        substitutions:
            The substitutions that target it.

        parameters:
            The parameter symbols of the system.
    """

    def __init__(
        self,
        path: str,
        patch: Patch,
        substitutions: Sequence[Substitution],
        parameters: Parameters,
    ) -> None:
        self.path = path
        self.max_size = int(patch.max_size)
        self.prototiles = tuple(patch.prototiles)
        if not self.prototiles:
            error_msg = f'Patch {path!r} has no prototiles. Give at least one.'
            raise RewritingError(error_msg)
        for prototile in self.prototiles:
            prototile.points()
        self.names = tuple(prototile.name for prototile in self.prototiles)
        self.counted = tuple(name for name in dict.fromkeys(self.names) if IDENTIFIER.match(name))
        self.parameters = parameters
        tiles = patch.tiles if patch.tiles is not None else [(self.names[0], None)]
        self.codes = np.array([self._code(name, 'the tiles') for name, _ in tiles], dtype=np.int64)
        self.placements = [placement for _, placement in tiles]
        self.pieces: dict[int, list[tuple[int, Any]]] = {}
        for substitution in substitutions:
            code = self._code(substitution.prototile, repr(substitution))
            if code in self.pieces:
                error_msg = f'Patch {path!r} has two substitutions of {substitution.prototile!r}. Keep one.'
                raise RewritingError(error_msg)
            self.pieces[code] = [(self._code(name, repr(substitution)), affine) for name, affine in substitution.pieces]

    def _code(self, name: str, where: str) -> int:
        """Return the code of a prototile by name."""
        if name not in self.names:
            error_msg = (
                f'{where} of patch {self.path!r} names the prototile {name!r}, which is not one of {self.names}.'
            )
            raise RewritingError(error_msg)
        return self.names.index(name)

    def _values(self, values: npt.NDArray[np.float64]) -> dict[Any, float]:
        """Return the parameter values by symbol."""
        return dict(zip(self.parameters.symbols, values.tolist(), strict=True))

    def initial(self, values: npt.NDArray[np.float64]) -> PatchStructure:
        """Return the patch at generation 0.

        Args:
            values:
                The parameter values.

        Returns:
            patch:
                The initial tiles.
        """
        by_symbol = self._values(values)
        matrices = [np.eye(3) if item is None else item.matrix_of(by_symbol) for item in self.placements]
        return PatchStructure(self.prototiles, self.codes, np.reshape(matrices, (-1, 3, 3)), 0)

    def counts(self, patch: PatchStructure) -> list[float]:
        """Return the count of every prototile and the number of tiles.

        Args:
            patch:
                The patch.

        Returns:
            counts:
                The counts, in the order of the states.
        """
        return [float(patch.count(name)) for name in self.counted] + [float(len(patch))]

    def rewrite(
        self,
        patch: PatchStructure,
        values: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> PatchStructure:
        """Return the patch of the next generation, every tile replaced by its pieces in parallel.

        Args:
            patch:
                The patch.

            values:
                The parameter values.

            rng:
                The random stream, unused since substitutions are deterministic.

        Returns:
            patch:
                The next patch.

        Raises:
            RewritingError:
                The next patch would exceed the maximum size.
        """
        del rng
        by_symbol = self._values(values)
        codes, matrices = patch.codes, patch.matrices
        children = np.ones(len(self.prototiles), dtype=np.int64)
        for code, pieces in self.pieces.items():
            children[code] = len(pieces)
        lengths = children[codes]
        size = int(lengths.sum())
        if size > self.max_size:
            error_msg = (
                f'Patch {self.path!r} would reach {size} tiles at generation {patch.generation + 1}, above its maximum '
                f'size {self.max_size}. Raise max_size or run fewer generations.'
            )
            raise RewritingError(error_msg)
        offsets = np.cumsum(lengths) - lengths
        new_codes = np.empty(size, dtype=np.int64)
        new_matrices = np.empty((size, 3, 3))
        kept = np.flatnonzero(~np.isin(codes, list(self.pieces)))
        new_codes[offsets[kept]] = codes[kept]
        new_matrices[offsets[kept]] = matrices[kept]
        for code, pieces in self.pieces.items():
            rows = np.flatnonzero(codes == code)
            if not len(rows):
                continue
            maps = np.stack([affine.matrix_of(by_symbol) for _, affine in pieces])
            places = offsets[rows][:, None] + np.arange(len(pieces))
            new_matrices[places] = np.einsum('tij,pjk->tpik', matrices[rows], maps)
            new_codes[places] = [piece for piece, _ in pieces]
        return PatchStructure(self.prototiles, new_codes, new_matrices, patch.generation + 1)


def _matches(attributes: dict[str, Any], pattern: dict[str, Any]) -> bool:
    """Check that a host node or edge has the label of a pattern node or edge, where the pattern gives one."""
    return 'label' not in pattern or attributes.get('label') == pattern['label']


class CompiledGraph:
    """Graph rewriting system of one host graph, with double-pushout productions matched by NetworkX.

    Args:
        path:
            The path of the host graph.

        host:
            The host graph element.

        productions:
            The graph productions that target it, applied one after the other within a generation.
    """

    def __init__(self, path: str, host: HostGraph, productions: Sequence[GraphProduction]) -> None:
        self.path = path
        self.graph = host.graph
        self.max_size = int(host.max_size)
        self.max_matches = int(host.max_matches)
        self.productions = tuple(productions)
        for production in self.productions:
            outside = set(production.interface.nodes) - (set(production.left.nodes) & set(production.right.nodes))
            if outside:
                error_msg = (
                    f'The interface of a production of {path!r} has the nodes {sorted(map(str, outside))}, which are '
                    'not nodes of both its left and its right graph.'
                )
                raise RewritingError(error_msg)
        labels = [label for _, label in self.graph.nodes(data='label')]
        labels += [label for item in self.productions for _, label in item.right.nodes(data='label')]
        self.counted = tuple(
            sorted({label for label in labels if isinstance(label, str) and IDENTIFIER.match(label)}),
        )

    def initial(self, values: npt.NDArray[np.float64]) -> GraphStructure:
        """Return the host graph at generation 0.

        Args:
            values:
                The parameter values, unused.

        Returns:
            graph:
                The initial graph.
        """
        del values
        return GraphStructure(self.graph, 0)

    def counts(self, graph: GraphStructure) -> list[float]:
        """Return the count of nodes of every label and the number of nodes.

        Args:
            graph:
                The graph.

        Returns:
            counts:
                The counts, in the order of the states.
        """
        return [float(graph.count(name)) for name in self.counted] + [float(len(graph))]

    def _key(self, graph: nx.Graph, first: Any, second: Any) -> Any:  # noqa: ANN401
        """Return the key of an edge, ordered for directed graphs."""
        return (first, second) if graph.is_directed() else frozenset((first, second))

    def _select(self, graph: nx.Graph, production: GraphProduction, rng: np.random.Generator) -> list[dict[Any, Any]]:
        """Return a maximal set of parallel independent matches that satisfy the dangling condition."""
        matcher_type = isomorphism.DiGraphMatcher if graph.is_directed() else isomorphism.GraphMatcher
        matcher = matcher_type(graph, production.left, node_match=_matches, edge_match=_matches)
        found = [
            {pattern: host for host, pattern in item.items()}
            for item in itertools.islice(matcher.subgraph_monomorphisms_iter(), self.max_matches)
        ]
        left, interface = production.left, production.interface
        used: set[Any] = set()
        deleted: set[Any] = set()
        chosen = []
        for index in rng.permutation(len(found)):
            match = found[index]
            edges = {(self._key(graph, match[u], match[v]), interface.has_edge(u, v)) for u, v in left.edges}
            nodes = {match[node] for node in left.nodes}
            removed_nodes = {match[node] for node in left.nodes if node not in interface}
            removed = removed_nodes | {key for key, kept in edges if not kept}
            items = nodes | {key for key, _ in edges}
            incident = {self._key(graph, u, v) for node in removed_nodes for u, v in nx.edges(graph, node)}
            if graph.is_directed():
                incident |= {self._key(graph, u, v) for node in removed_nodes for u, v in graph.in_edges(node)}
            if not incident <= removed or items & deleted or removed & used:
                continue
            used |= items
            deleted |= removed
            chosen.append(match)
        return chosen

    def _apply(self, graph: nx.Graph, production: GraphProduction, matches: list[dict[Any, Any]]) -> None:
        """Delete and glue the parts of every chosen match."""
        left, interface, right = production.left, production.interface, production.right
        for match in matches:
            for u, v in left.edges:
                if not interface.has_edge(u, v) and graph.has_edge(match[u], match[v]):
                    graph.remove_edge(match[u], match[v])
            graph.remove_nodes_from([match[node] for node in left.nodes if node not in interface])
        fresh = itertools.count(max((node for node in graph.nodes if isinstance(node, int)), default=-1) + 1)
        for match in matches:
            names = {node: match[node] for node in interface.nodes}
            for node in right.nodes:
                if node not in names:
                    names[node] = next(item for item in fresh if not graph.has_node(item))
                graph.add_node(names[node], **right.nodes[node])
            for u, v, attributes in right.edges(data=True):
                if not interface.has_edge(u, v):
                    graph.add_edge(names[u], names[v], **attributes)

    def rewrite(
        self,
        structure: GraphStructure,
        values: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> GraphStructure:
        """Return the host graph of the next generation.

        Args:
            structure:
                The host graph.

            values:
                The parameter values, unused.

            rng:
                The random stream that orders the matches.

        Returns:
            graph:
                The next host graph.

        Raises:
            RewritingError:
                The next graph would exceed the maximum size.
        """
        del values
        graph = nx.DiGraph(structure.graph) if structure.graph.is_directed() else nx.Graph(structure.graph)
        for production in self.productions:
            self._apply(graph, production, self._select(graph, production, rng))
            if graph.number_of_nodes() > self.max_size:
                error_msg = (
                    f'Host graph {self.path!r} would reach {graph.number_of_nodes()} nodes at generation '
                    f'{structure.generation + 1}, above its maximum size {self.max_size}.'
                )
                raise RewritingError(error_msg)
        return GraphStructure(graph, structure.generation + 1)


type Compiled = CompiledWord | CompiledPatch | CompiledGraph


def is_rewriting(system: System) -> bool:
    """Check whether a system holds a word, a patch, or a host graph.

    Args:
        system:
            The system.

    Returns:
        found:
            Whether one of its elements, at any depth, is rewritten.
    """
    return any(isinstance(element, REWRITTEN) for element in system.elements) or any(
        is_rewriting(child) for child in system.systems
    )


def _parameters(nodes: list[Node]) -> tuple[list[Entry], list[float], dict[str, Symbols], dict[str, Any]]:
    """Return the parameter entries and values of the rewriting interactions, with their namespaces and symbols."""
    entries: list[Entry] = []
    values: list[float] = []
    namespaces: dict[str, Symbols] = {}
    symbols: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node.component, Rewriting):
            continue
        names = {}
        for name, declaration in declarations_of(type(node.component), VariableKind.PARAMETER).items():
            value = node.value_of(name)
            if np.ndim(value) or not declaration.domain.contains(value):
                error_msg = f'{node.path}.{name} = {value!r} is not a number in the {declaration.domain.value} domain.'
                raise ValidationError(error_msg)
            dotted = f'{node.path}.{name}'
            names[name] = symbol(f'{name}_{node.path}'.replace('/', '_').replace('-', '_'))
            symbols[dotted] = names[name]
            entries.append(Entry(dotted, len(entries), (), declaration.domain))
            values.append(float(value))
        namespaces[node.path] = Symbols(**names, t=symbol('t'))
    return entries, values, namespaces, symbols


def _compiled(
    node: Node,
    laws: list[RewritingLaw],
    parameters: Parameters,
    values: npt.NDArray[np.float64],
) -> Compiled:
    """Return the compiled rewriting system of one element from the laws that target it."""
    element = node.component
    kinds = {Word: Production, Patch: Substitution, HostGraph: GraphProduction}
    expected = next(kind for owner, kind in kinds.items() if isinstance(element, owner))
    wrong = [law for law in laws if not isinstance(law, expected)]
    if wrong:
        error_msg = (
            f'{type(element).__name__} {node.path!r} is rewritten by {expected.__name__} laws, but got {wrong[0]!r}.'
        )
        raise RewritingError(error_msg)
    if isinstance(element, Word):
        return CompiledWord(
            node.path,
            element,
            [law for law in laws if isinstance(law, Production)],
            parameters,
            values,
        )
    if isinstance(element, Patch):
        return CompiledPatch(node.path, element, [law for law in laws if isinstance(law, Substitution)], parameters)
    return CompiledGraph(node.path, element, [law for law in laws if isinstance(law, GraphProduction)])


def compile_rewriting(system: System) -> Problem:
    """Compile a system of words, patches, or host graphs and the interactions that rewrite them.

    Args:
        system:
            The system.

    Returns:
        problem:
            The problem of the rewriting form, whose states are the counts and sizes of the structures.

    Raises:
        IncompatibleLawsError:
            The system has elements or interactions that do not take part in rewriting.

        RoleError:
            A rewriting interaction targets a path that is not a word, a patch, or a host graph.

        RewritingError:
            A production, substitution, or graph production is malformed.

    Examples:
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import Production, Rewrite, System, Word
        >>> system = System(
        ...     components=[Word('algae', axiom='A')],
        ...     interactions=[Rewrite('rules', target='algae', rules=[Production('A', 'AB'), Production('B', 'A')])],
        ... )
        >>> problem = compile_system(system)
        >>> problem.form.value, problem.states.names
        ('rewriting system', ['algae.A', 'algae.B', 'algae.size'])
    """
    nodes = flatten_system(system)
    elements = {node.path: node for node in nodes if isinstance(node.component, REWRITTEN)}
    for node in nodes:
        if isinstance(node.component, Group) and issubclass(node.component.element, REWRITTEN):
            error_msg = f'{node.path} is a group of rewritten elements. Declare one element per structure.'
            raise RewritingError(error_msg)
        if node.path not in elements and not isinstance(node.component, Rewriting):
            error_msg = (
                f'{node.path} takes no part in rewriting, but its system holds words, patches, or host graphs. A '
                'rewriting system holds only rewritten elements and the interactions that rewrite them.'
            )
            raise IncompatibleLawsError(error_msg)
    entries, values, namespaces, symbols = _parameters(nodes)
    parameters = Parameters(tuple(symbols.values()), symbol('t'))
    initial_parameters = np.asarray(values, dtype=float)
    laws: dict[str, list[RewritingLaw]] = {path: [] for path in elements}
    interactions = []
    for node in nodes:
        if not isinstance(node.component, Rewriting):
            continue
        target = f'{node.scope}{node.component.target}'
        if target not in elements:
            error_msg = (
                f'{node.path} rewrites {target!r}, which is not a word, a patch, or a host graph. Rewritten elements '
                f'are: {", ".join(elements) or "none"}.'
            )
            raise RoleError(error_msg)
        laws[target] += list(node.component.productions(namespaces[node.path]))
        interactions.append(CompiledInteraction(node.path, node.component, {'target': target}))
    compiled = tuple(_compiled(elements[path], laws[path], parameters, initial_parameters) for path in elements)
    states: list[Entry] = []
    initial: list[float] = []
    compiled_elements = []
    for item in compiled:
        names = [*item.counted, 'size']
        component = cast('Element', elements[item.path].component)
        compiled_elements.append(
            CompiledElement(
                path=item.path,
                element=component,
                states={name: f'{item.path}.{name}' for name in names},
                parameters={},
                visual=component.visual,
            ),
        )
        for name in names:
            states.append(Entry(f'{item.path}.{name}', len(states), (), Domain.NONNEGATIVE_INTEGER))
        initial += item.counts(item.initial(initial_parameters))  # type: ignore[arg-type]
    return Problem(
        form=Form.REWRITING,
        states=Layout(states),
        parameters=Layout(entries),
        initial_state=np.asarray(initial, dtype=float),
        initial_parameters=initial_parameters,
        time=symbol('t'),
        symbols=symbols,
        equations={},
        equation_symbols={},
        has_callables=False,
        has_group_sums=False,
        elements=tuple(compiled_elements),
        interactions=tuple(interactions),
        system=system,
        rewriting=compiled,
    )
