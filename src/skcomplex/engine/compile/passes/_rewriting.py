"""Compile rewriting systems: words, patches, and host graphs with their productions, as problems of generations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.algorithms import isomorphism
from sage.all__sagemath_symbolics import NN

from ....core.elements import Element, GroupElement, HostGraphElement, PatchElement, WordElement
from ....core.errors import IncompatibleLawsError, RewritingError, ValidationError
from ....core.laws import GraphProductionLaw, ProductionLaw, SubstitutionLaw
from ....core.system import System
from ...problems import (
    BaseProblem,
    CompiledElement,
    Entry,
    Form,
    Layout,
    VariableKind,
    contains,
    declarations_of,
    symbol,
)
from .._rewriting_words import IDENTIFIER, CompiledWord, Parameters
from .._structures import GraphStructure, PatchStructure, affine_matrix
from .._system_compilers import BaseSystemCompiler
from .._tree import Node, flatten_system
from ._problem import visual_of

LAWS_OF = {'sequence': ProductionLaw, 'tiling': SubstitutionLaw, 'graph': GraphProductionLaw}
type Compiled = CompiledWord | CompiledPatch | CompiledGraph

PRODUCTION_LAWS = (ProductionLaw, SubstitutionLaw, GraphProductionLaw)
STRUCTURES = {WordElement: 'sequence', PatchElement: 'tiling', HostGraphElement: 'graph'}
type Production = ProductionLaw | SubstitutionLaw | GraphProductionLaw


def structure_of(element_type: type) -> str | None:
    """Return the kind of structure an element type rewrites, or `None` for a type that rewrites none."""
    if not isinstance(element_type, type):
        return None
    return next((kind for base, kind in STRUCTURES.items() if issubclass(element_type, base)), None)


def _matches(attributes: dict[str, Any], pattern: dict[str, Any]) -> bool:
    """Check that a host node or edge has the label of a pattern node or edge, where the pattern gives one."""
    return 'label' not in pattern or attributes.get('label') == pattern['label']


def _type_of(component: object) -> type:
    """Return the element type of a group, or the type of any other component."""
    return component.element if isinstance(component, GroupElement) else type(component)


def _parameters(nodes: list[Node]) -> tuple[list[Entry], list[float], dict[str, SimpleNamespace], dict[str, Any]]:
    """Return the parameter entries and values of the rewritten elements, with their namespaces and symbols."""
    entries: list[Entry] = []
    values: list[float] = []
    namespaces: dict[str, SimpleNamespace] = {}
    symbols: dict[str, Any] = {}
    for node in nodes:
        names = {}
        for name, declaration in declarations_of(type(node.component), VariableKind.PARAMETER).items():
            value = node.value_of(name)
            if np.ndim(value) or not contains(declaration.domain, value):
                error_msg = f'{node.path}.{name} = {value!r} is not a number in the {declaration.domain.value} domain.'
                raise ValidationError(error_msg)
            dotted = f'{node.path}.{name}'
            names[name] = symbol(f'{name}_{node.path}'.replace('/', '_').replace('-', '_'))
            symbols[dotted] = names[name]
            entries.append(Entry(dotted, len(entries), (), declaration.domain))
            values.append(float(value))
        namespaces[node.path] = SimpleNamespace(**names, t=symbol('t'))
    return entries, values, namespaces, symbols


def _productions(node: Node, namespace: SimpleNamespace) -> list[Production]:
    """Return the checked productions, substitutions, or graph productions a word, a patch, or a host graph returns."""
    laws = node.component.run(namespace)
    wrong = [law for law in laws if not isinstance(law, PRODUCTION_LAWS)]
    if wrong:
        error_msg = (
            f'{node.path} returns {type(wrong[0]).__name__} from `_get_laws`, but a word, a patch, or a host graph '
            'changes by its productions only. Return ProductionLaw, SubstitutionLaw, or GraphProductionLaw laws.'
        )
        raise RewritingError(error_msg)
    return [law.check() for law in laws]


def _compiled(
    node: Node,
    laws: list[Production],
    parameters: Parameters,
    values: npt.NDArray[np.float64],
) -> Compiled:
    """Return the compiled rewriting system of one element from the laws that target it."""
    element = node.component
    kind = structure_of(type(element))
    if kind is None:
        error_msg = f'{node.path} holds no sequence, tiling, or graph that rewriting laws rewrite.'
        raise ValidationError(error_msg)
    expected = LAWS_OF[kind]
    wrong = [law for law in laws if not isinstance(law, expected)]
    if wrong:
        error_msg = (
            f'{type(element).__name__} {node.path!r} is rewritten by {expected.__name__}s, but got {wrong[0]!r}.'
        )
        raise RewritingError(error_msg)
    if kind == 'sequence':
        return CompiledWord(
            node.path,
            element,
            [law for law in laws if isinstance(law, ProductionLaw)],
            parameters,
            values,
        )
    if kind == 'tiling':
        return CompiledPatch(node.path, element, [law for law in laws if isinstance(law, SubstitutionLaw)], parameters)
    return CompiledGraph(node.path, element, [law for law in laws if isinstance(law, GraphProductionLaw)])


def is_rewriting(system: System) -> bool:
    """Check whether a system holds a word, a patch, or a host graph.

    Args:
        system:
            The system.

    Returns:
        found:
            Whether one of its elements, at any depth, is rewritten.
    """
    return any(structure_of(_type_of(element)) for element in system.elements) or any(
        is_rewriting(child) for child in system.systems
    )


def compile_rewriting(system: System) -> 'RewritingProblem':
    """Compile a system of words, patches, or host graphs, each rewritten by the rewriting laws it returns.

    Args:
        system:
            The system.

    Returns:
        problem:
            The problem of the rewriting form, whose states are the counts and sizes of the structures.

    Raises:
        IncompatibleLawsError:
            The system has elements or interactions that do not take part in rewriting.

        RewritingError:
            A rewritten element returns a law that is not a rewriting law, or a production, substitution, or graph
            production is malformed.

    Examples:
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.laws import ProductionLaw
        >>> from skcomplex.core.elements import WordElement
        >>> class AlgaeElement(WordElement):
        ...     def _get_laws(self, namespace):
        ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
        >>> problem = compile_system(System(elements=[AlgaeElement('algae', axiom='A')], time=DiscreteTime()))
        >>> problem.form.value, problem.states.names
        ('rewriting system', ['algae.A', 'algae.B', 'algae.size'])
    """
    nodes = flatten_system(system)
    elements = {node.path: node for node in nodes if structure_of(_type_of(node.component))}
    for node in nodes:
        if isinstance(node.component, GroupElement) and structure_of(node.component.element):
            error_msg = f'{node.path} is a group of rewritten elements. Declare one element per structure.'
            raise RewritingError(error_msg)
        if node.path not in elements:
            error_msg = (
                f'{node.path} takes no part in rewriting, but its system holds words, patches, or host graphs. A '
                'rewriting system holds only rewritten elements, which rewrite themselves by their laws.'
            )
            raise IncompatibleLawsError(error_msg)
    entries, values, namespaces, symbols = _parameters(nodes)
    parameters = Parameters(tuple(symbols.values()), symbol('t'))
    initial_parameters = np.asarray(values, dtype=float)
    compiled = tuple(
        _compiled(node, _productions(node, namespaces[path]), parameters, initial_parameters)
        for path, node in elements.items()
    )
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
                parameters={name: f'{item.path}.{name}' for name in vars(namespaces[item.path]) if name != 't'},
                visual=visual_of(elements[item.path]),
            ),
        )
        for name in names:
            states.append(Entry(f'{item.path}.{name}', len(states), (), NN))
        initial += item.counts(item.initial(initial_parameters))  # type: ignore[arg-type]
    return RewritingProblem(
        form=Form.REWRITING,
        states=Layout(states),
        parameters=Layout(entries),
        initial_state=np.asarray(initial, dtype=float),
        initial_parameters=initial_parameters,
        time=symbol('t'),
        symbols=symbols,
        elements=tuple(compiled_elements),
        system=system,
        rewriting=compiled,
    )


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
        patch: PatchElement,
        substitutions: Sequence[SubstitutionLaw],
        parameters: Parameters,
    ) -> None:
        self.path = path
        self.max_size = int(patch.max_size)
        self.prototiles = tuple(patch.prototiles)
        if not self.prototiles:
            error_msg = f'Patch {path!r} has no prototiles. Give at least one.'
            raise RewritingError(error_msg)
        for prototile in self.prototiles:
            prototile.check()
        self.names = tuple(prototile.name for prototile in self.prototiles)
        self.counted = tuple(name for name in dict.fromkeys(self.names) if IDENTIFIER.match(name))
        self.parameters = parameters
        tiles = patch.tiles if patch.tiles is not None else [(self.names[0], None)]
        self.codes = np.array([self._code(name, 'the tiles') for name, _ in tiles], dtype=np.int64)
        self.placements = [placement for _, placement in tiles]
        self.pieces: dict[int, list[tuple[int, Any]]] = {}
        for substitution in substitutions:
            where = f'The substitution of {substitution.prototile!r}'
            code = self._code(substitution.prototile, where)
            if code in self.pieces:
                error_msg = f'Patch {path!r} has two substitutions of {substitution.prototile!r}. Keep one.'
                raise RewritingError(error_msg)
            self.pieces[code] = [(self._code(name, where), affine) for name, affine in substitution.pieces]

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
        matrices = [np.eye(3) if item is None else affine_matrix(item, by_symbol) for item in self.placements]
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
        return [float(patch.count(name)) for name in self.counted] + [float(patch.parts)]

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
            maps = np.stack([affine_matrix(affine, by_symbol) for _, affine in pieces])
            places = offsets[rows][:, None] + np.arange(len(pieces))
            new_matrices[places] = np.einsum('tij,pjk->tpik', matrices[rows], maps)
            new_codes[places] = [piece for piece, _ in pieces]
        return PatchStructure(self.prototiles, new_codes, new_matrices, patch.generation + 1)


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

    def __init__(self, path: str, host: HostGraphElement, productions: Sequence[GraphProductionLaw]) -> None:
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
        return [float(graph.count(name)) for name in self.counted] + [float(graph.parts)]

    def _key(self, graph: nx.Graph, first: Any, second: Any) -> Any:  # noqa: ANN401
        """Return the key of an edge, ordered for directed graphs."""
        return (first, second) if graph.is_directed() else frozenset((first, second))

    def _select(
        self,
        graph: nx.Graph,
        production: GraphProductionLaw,
        rng: np.random.Generator,
    ) -> list[dict[Any, Any]]:
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

    def _apply(self, graph: nx.Graph, production: GraphProductionLaw, matches: list[dict[Any, Any]]) -> None:
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


@dataclass(frozen=True)
class RewritingProblem(BaseProblem):
    """Problem of the rewriting form, with the compiled rewriting system of every rewritten element.

    Args:
        rewriting:
            The compiled rewriting systems of the words, patches, and host graphs.
    """

    rewriting: tuple[Compiled, ...] = field(default=(), repr=False)


class RewritingCompiler(BaseSystemCompiler):
    """Pass that compiles a system of words, patches, or host graphs into a problem of the rewriting form."""

    def _prepare(self, system: System) -> None:
        """Set `claims_`, whether a system holds a word, a patch, or a host graph.

        Args:
            system:
                The system.
        """
        self.claims_ = is_rewriting(system)

    def _run(
        self,
        system: System,
        *,
        backend: object,
        solver: object,
        initial_state: Mapping[str, object],
    ) -> BaseProblem:
        """Compile a rewriting system into a problem of the rewriting form.

        Args:
            system:
                The system, with the observables of the run.

            backend:
                The backend requested for the problem, which the rewriting form does not use.

            solver:
                The solver the problem runs on, which the rewriting form does not use.

            initial_state:
                Initial values by dotted state name, which the rewriting form does not use.

        Returns:
            problem:
                The problem of the rewriting form.
        """
        del backend, solver, initial_state
        return compile_rewriting(system)
