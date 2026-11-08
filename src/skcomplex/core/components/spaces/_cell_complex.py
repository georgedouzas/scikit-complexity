"""Define cell complexes.

A cell complex is a network of the sites, links, and faces of a lattice or a network with oriented incidence.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.sparse
import xgi

from ...errors import SpaceError, ValidationError
from ...specs import Tags
from ...types import Array, Indices
from ._lattice import LatticeSpace
from ._network import NetworkSpace

FACES = ('hyperedges', 'cliques')
LOWEST_DIMENSION = 2
CUBE_DIMENSION = 3
TRIANGLE_NODES = 3
PLAQUETTE_ROWS = 4
CUBE_ROWS = 6


class CellComplexSpace(NetworkSpace):
    """Network of the cells of a lattice or a network, with their oriented incidence.

    The incidence is between consecutive degrees. Its nodes are the sites or nodes of the space it derives from, its
    edges the oriented links or edges, and its hyperedges the oriented plaquettes or faces in cyclic node order. Every
    group, binding, and neighbourhood of a network space applies to it, so groups on its edges and hyperedges hold
    states on links and plaquettes. It has no layers.

    The links of a lattice are numbered by direction and then by site, and point along the positive direction of their
    axis. The plaquettes are numbered by plane and then by site, with planes `(0, 1), (0, 2), ...` in increasing order,
    and turn from the first axis of the plane to the second. Fixed and reflecting edges drop the cells that leave the
    box. A face of a network is oriented by its node cycle. The faces of a clique complex and the triangles of
    simplicial hyperedges take their incidence from `xgi.linalg.hodge_matrix.boundary_matrix`, and the other faces from
    the sum of the oriented edges around them.

    Args:
        space:
            The `LatticeSpace` or the `NetworkSpace` the cells derive from.

        faces:
            The faces of a network: `None` for a minimum cycle basis, `'hyperedges'` for the hyperedges of the network,
            `'cliques'` for its cliques up to `dimension`, or a sequence of node cycles. A lattice takes `None`.

        dimension:
            The highest degree of the cells. Cells of degree 3 take part in incidence and topology only, since a group
            lives on sites, edges, or hyperedges.

        label:
            The label of the space in its system.

    Attributes:
        incidences_:
            The oriented incidence of every degree `k` as a SciPy CSR array with entries of 1 and -1, with one row per
            cell of degree `k + 1` and one column per cell of degree `k`, so `incidences_[k + 1] @ incidences_[k]` is 0.

        cofaces_:
            The cells of degree `k + 1` that contain every cell of degree `k`, with one row per cell, padded with the
            number of cells of degree `k + 1`.

        orientations_:
            The sign of every coface of `cofaces_` relative to its cell, padded with 0.

        boundaries_:
            The edges around every face in the cyclic order of its nodes, from the edge that leaves its first node, with
            one row per face, padded with -1.

        boundary_signs_:
            The sign of the traversal of every edge of `boundaries_`, 1 along its orientation and -1 against it, padded
            with 0, so the path-ordered product of link matrices along a row is the holonomy of the face.

        stars_:
            The diagonal of the Hodge star of every degree, the ratio of dual to primal cell volumes on a lattice and 1
            on a network.

        link_axes_:
            The axis of every link of a lattice, empty on a network.

        face_planes_:
            The two axes of the plane of every plaquette of a lattice, empty on a network.

    Raises:
        SpaceError:
            A face uses a pair of nodes that the graph does not join, the network has a loop, or the lattice is not
            square.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace, NetworkSpace
        >>> square = CellComplexSpace(LatticeSpace((8, 8))).prepare()
        >>> [matrix.shape for matrix in square.incidences_]
        [(128, 64), (64, 128)]
        >>> int((square.incidences_[1] @ square.incidences_[0]).count_nonzero())
        0
        >>> chorded = nx.cycle_graph(4)
        >>> chorded.add_edge(0, 2)
        >>> CellComplexSpace(NetworkSpace(chorded)).prepare().hyperedges_
        ((0, 1, 2), (0, 2, 3))
        >>> square.boundaries_[0].tolist(), square.boundary_signs_[0].tolist()
        ([0, 72, 1, 64], [1.0, 1.0, -1.0, -1.0])
    """

    space_tags: ClassVar[tuple[str, ...]] = ('cell complex',)

    def _get_tags(self) -> Tags:
        """Return the tags of the cell complex, and no laws."""
        return Tags(space=frozenset(self.space_tags))

    def __init__(
        self,
        space: LatticeSpace | NetworkSpace,
        *,
        faces: str | Sequence[Sequence[object]] | None = None,
        dimension: int = LOWEST_DIMENSION,
        label: str = 'space',
    ) -> None:
        super().__init__(nx.Graph(), label=label)
        self.space = space
        self.faces = faces
        self.dimension = dimension

    def _lattice_links(self, place: LatticeSpace) -> tuple[Indices, Indices, Indices]:
        """Return the source and target of every link of a lattice, and the link table.

        The table holds the link of every axis and site, or -1.
        """
        sites = np.arange(place.size_).reshape(place.shape_)
        sources, targets = [], []
        table = np.full((place.n_dim_, place.size_), -1, dtype=np.intp)
        count = 0
        for axis, length in enumerate(place.shape_):
            kept = np.ones(place.shape_, dtype=bool)
            if place._axis_boundary(axis) != 'periodic':
                kept = np.indices(place.shape_)[axis] < length - 1
            sources.append(sites[kept])
            targets.append(np.roll(sites, -1, axis=axis)[kept])
            table[axis, sites[kept]] = count + np.arange(sites[kept].size)
            count += sites[kept].size
        return np.concatenate(sources).astype(np.intp), np.concatenate(targets).astype(np.intp), table

    def _lattice_steps(self, place: LatticeSpace) -> Indices:
        """Return the site one step along every axis from every site.

        The steps wrap periodically over the shape.
        """
        sites = np.arange(place.size_).reshape(place.shape_)
        return np.stack([np.roll(sites, -1, axis=axis).ravel() for axis in range(place.n_dim_)])

    def _sparse(self, rows: Indices, columns: Indices, signs: Array, shape: tuple[int, int]) -> scipy.sparse.csr_array:
        """Return the CSR array of the oriented entries.

        Entries that repeat add up and may cancel.
        """
        matrix = scipy.sparse.coo_array((signs, (rows, columns)), shape=shape).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix

    def _edge_incidence(self, sources: Indices, targets: Indices) -> scipy.sparse.csr_array:
        """Return the incidence of sites and links.

        It is -1 at the source of every link and 1 at its target.
        """
        links = np.arange(sources.size, dtype=np.intp)
        signs = np.concatenate([-np.ones(links.size), np.ones(links.size)])
        rows, columns = np.concatenate([links, links]), np.concatenate([sources, targets])
        return self._sparse(rows, columns, signs, (links.size, self.graph.number_of_nodes()))

    def _lattice_plaquettes(
        self,
        place: LatticeSpace,
        links: Indices,
    ) -> tuple[Indices, scipy.sparse.csr_array, Indices, Indices]:
        """Return the node cycle, the incidence, and the plaquette tables of a lattice.

        The tables hold the plaquette of every plane and the links around every plaquette.
        """
        steps = self._lattice_steps(place)
        cycles, rows, columns, signs, boundaries = [], [], [], [], []
        planes = list(itertools.combinations(range(place.n_dim_), 2))
        table = np.full((len(planes), place.size_), -1, dtype=np.intp)
        count = 0
        for number, (first, second) in enumerate(planes):
            sites = np.arange(place.size_, dtype=np.intp)
            edges = (
                links[first, sites],
                links[second, steps[first]],
                links[first, steps[second]],
                links[second, sites],
            )
            kept = sites[np.all(np.stack(edges) >= 0, axis=0)]
            cycle = (kept, steps[first, kept], steps[second, steps[first, kept]], steps[second, kept])
            cycles.append(np.stack(cycle, axis=1))
            boundaries.append(np.stack([edge[kept] for edge in edges], axis=1))
            table[number, kept] = count + np.arange(kept.size)
            for edge, sign in zip(edges, (1.0, 1.0, -1.0, -1.0), strict=True):
                rows.append(table[number, kept])
                columns.append(edge[kept])
                signs.append(np.full(kept.size, sign))
            count += kept.size
        shape = (count, int((links >= 0).sum()))
        matrix = self._sparse(np.concatenate(rows), np.concatenate(columns), np.concatenate(signs), shape)
        found = np.concatenate(boundaries).reshape(-1, PLAQUETTE_ROWS)
        return np.concatenate(cycles).reshape(-1, PLAQUETTE_ROWS), matrix, table, found

    def _lattice_cubes(self, place: LatticeSpace, plaquettes: Indices) -> scipy.sparse.csr_array:
        """Return the incidence of the plaquettes and the cubes of a lattice.

        The lattice has at least three dimensions.
        """
        steps = self._lattice_steps(place)
        planes = list(itertools.combinations(range(place.n_dim_), 2))
        rows, columns, signs = [], [], []
        count = 0
        for first, second, third in itertools.combinations(range(place.n_dim_), 3):
            bottom = plaquettes[planes.index((first, second))]
            middle = plaquettes[planes.index((first, third))]
            side = plaquettes[planes.index((second, third))]
            sites = np.arange(place.size_, dtype=np.intp)
            cells = (
                (bottom[sites], -1.0),
                (bottom[steps[third]], 1.0),
                (middle[sites], 1.0),
                (middle[steps[second]], -1.0),
                (side[sites], -1.0),
                (side[steps[first]], 1.0),
            )
            kept = sites[np.all(np.stack([cell for cell, _ in cells]) >= 0, axis=0)]
            for cell, sign in cells:
                rows.append(count + np.arange(kept.size))
                columns.append(cell[kept])
                signs.append(np.full(kept.size, sign))
            count += kept.size
        shape = (count, int((plaquettes >= 0).sum()))
        if not rows:
            return self._sparse(np.empty(0, np.intp), np.empty(0, np.intp), np.empty(0), shape)
        return self._sparse(np.concatenate(rows), np.concatenate(columns), np.concatenate(signs), shape)

    def _prepare_lattice(self, place: LatticeSpace) -> None:
        """Set the graph, the plaquettes, and the incidence of the cells of a lattice."""
        sources, targets, links = self._lattice_links(place)
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(place.size_))
        graph.add_edges_from(zip(sources.tolist(), targets.tolist(), strict=True))
        nx.set_node_attributes(graph, dict(enumerate(place.positions_)), 'pos')
        self.graph = graph
        cycles, faces, plaquettes, boundaries = self._lattice_plaquettes(place, links)
        self.hyperedges = [tuple(cycle) for cycle in cycles.tolist()]
        self.boundaries_ = boundaries
        self.boundary_signs_ = np.tile(np.asarray([1.0, 1.0, -1.0, -1.0]), (boundaries.shape[0], 1))
        self.incidences_: tuple[scipy.sparse.csr_array, ...] = (self._edge_incidence(sources, targets), faces)
        if self.dimension >= CUBE_DIMENSION and place.n_dim_ >= CUBE_DIMENSION:
            self.incidences_ = (*self.incidences_, self._lattice_cubes(place, plaquettes))
        self.stars_ = self._lattice_stars(place, links, plaquettes)
        self.link_axes_ = self._link_axes(links)
        self.face_planes_ = self._face_planes(place, plaquettes)

    def _link_axes(self, links: Indices) -> Indices:
        """Return the axis of every link.

        It comes from the table of the link of every axis and site.
        """
        axes, sites = np.nonzero(links >= 0)
        found = np.empty(axes.size, dtype=np.intp)
        found[links[axes, sites]] = axes
        return found

    def _face_planes(self, place: LatticeSpace, plaquettes: Indices) -> Indices:
        """Return the two axes of every plaquette.

        They come from the table of the plaquette of every plane and site.
        """
        planes = np.asarray(list(itertools.combinations(range(place.n_dim_), 2)), dtype=np.intp).reshape(-1, 2)
        numbers, sites = np.nonzero(plaquettes >= 0)
        found = np.empty((numbers.size, 2), dtype=np.intp)
        found[plaquettes[numbers, sites]] = planes[numbers]
        return found

    def _lattice_stars(self, place: LatticeSpace, links: Indices, plaquettes: Indices) -> tuple[Array, ...]:
        """Return the ratio of dual to primal cell volumes of the cells of a lattice.

        The cells are the sites, the links, and the plaquettes.
        """
        spacings = np.asarray(place.spacings_)
        total = float(np.prod(spacings))
        stars = [np.full(place.size_, total), np.empty(int((links >= 0).sum()))]
        for axis, length in enumerate(spacings):
            stars[1][links[axis][links[axis] >= 0]] = total / length**2
        faces = np.empty(int((plaquettes >= 0).sum()))
        for number, plane in enumerate(itertools.combinations(range(place.n_dim_), 2)):
            cells = plaquettes[number][plaquettes[number] >= 0]
            faces[cells] = total / float(np.prod(spacings[list(plane)])) ** 2
        return (*stars, faces)

    def _cycle_of(self, nodes: list[object]) -> list[object]:
        """Return the nodes of a cycle of a minimum cycle basis in cyclic order.

        The order starts from the first node in node order.
        """
        order = {node: number for number, node in enumerate(self.graph.nodes)}
        subgraph = nx.Graph(self.graph).subgraph(nodes)
        start = min(nodes, key=lambda node: order[node])
        cycle, previous, current = [start], start, min(subgraph.neighbors(start), key=lambda node: order[node])
        while current != start:
            cycle.append(current)
            onward = [node for node in subgraph.neighbors(current) if node != previous]
            previous, current = current, min(onward, key=lambda node: order[node])
        return cycle

    def _network_faces(self, place: NetworkSpace) -> list[tuple[object, ...]]:
        """Return the node cycle of every face of a network.

        For cliques it is the nodes of every triangle in node order.
        """
        if self.faces is None:
            bases = nx.minimum_cycle_basis(nx.Graph(self.graph))
            ordered = [self._cycle_of(list(cycle)) for cycle in bases]
            order = {node: number for number, node in enumerate(self.graph.nodes)}
            return sorted((tuple(cycle) for cycle in ordered), key=lambda cycle: [order[node] for node in cycle])
        if self.faces == 'hyperedges':
            return place._get_hyperedges()
        if self.faces == 'cliques':
            return [tuple(nodes) for nodes in self._cliques(TRIANGLE_NODES)]
        return [tuple(cycle) for cycle in self.faces]

    def _cliques(self, size: int) -> list[list[object]]:
        """Return the cliques of a number of nodes, each in node order.

        The cliques come in lexicographic order of their nodes.
        """
        order = {node: number for number, node in enumerate(self.graph.nodes)}
        found = [
            sorted(clique, key=lambda node: order[node])
            for clique in nx.enumerate_all_cliques(nx.Graph(self.graph))
            if len(clique) == size
        ]
        return sorted(found, key=lambda clique: [order[node] for node in clique])

    def _check_faces(self, faces: list[tuple[object, ...]]) -> None:
        """Raise an error for a face of too few nodes, or with an unjoined pair.

        The unjoined pair is a pair the graph does not join.
        """
        for face in faces:
            if len(face) < TRIANGLE_NODES:
                error_msg = f'The face {face} has {len(face)} nodes. A face has at least {TRIANGLE_NODES} nodes.'
                raise SpaceError(error_msg)
            for first, second in zip(face, (*face[1:], face[0]), strict=True):
                if not self.graph.has_edge(first, second) and not self.graph.has_edge(second, first):
                    error_msg = (
                        f'The face {face} uses the edge ({first}, {second}), which the graph does not have. Give a '
                        'node cycle along the edges of the graph.'
                    )
                    raise SpaceError(error_msg)

    def _is_simplicial(self, faces: list[tuple[object, ...]]) -> bool:
        """Check whether the faces are the triangles of a simplicial complex.

        XGI builds the incidence of a simplicial complex.
        """
        if self.faces == 'cliques':
            return True
        return self.faces == 'hyperedges' and all(len(face) == TRIANGLE_NODES for face in faces)

    def _cycle_incidence(
        self,
        faces: list[tuple[object, ...]],
        sources: Indices,
        targets: Indices,
    ) -> scipy.sparse.csr_array:
        """Return the incidence of the edges and the faces.

        It comes from the sum of the oriented edges around every face.
        """
        index = {node: number for number, node in enumerate(self.graph.nodes)}
        ends = zip(sources.tolist(), targets.tolist(), strict=True)
        edges = {(first, second): edge for edge, (first, second) in enumerate(ends)}
        rows, columns, signs = [], [], []
        for row, face in enumerate(faces):
            nodes = [index[node] for node in face]
            for first, second in zip(nodes, (*nodes[1:], nodes[0]), strict=True):
                forward = (first, second) in edges
                rows.append(row)
                columns.append(edges[(first, second)] if forward else edges[(second, first)])
                signs.append(1.0 if forward else -1.0)
        shape = (len(faces), sources.size)
        return self._sparse(
            np.asarray(rows, dtype=np.intp),
            np.asarray(columns, dtype=np.intp),
            np.asarray(signs, dtype=float),
            shape,
        )

    def _parity(self, nodes: Sequence[int]) -> int:
        """Return the parity of an order of nodes relative to their sorted order.

        It is 0 for an even permutation and 1 for an odd one.
        """
        inversions = sum(first > second for first, second in itertools.combinations(nodes, 2))
        return inversions % 2

    def _simplicial_incidences(
        self,
        faces: list[tuple[object, ...]],
        sources: Indices,
        targets: Indices,
    ) -> tuple[scipy.sparse.csr_array, ...]:
        """Return the incidence of every order above the sites.

        It comes from the boundary matrices of XGI.
        """
        index = {node: number for number, node in enumerate(self.graph.nodes)}
        cells: list[list[tuple[int, ...]]] = [[(node,) for node in range(self.graph.number_of_nodes())]]
        cells.append(list(zip(sources.tolist(), targets.tolist(), strict=True)))
        cells.append([tuple(index[node] for node in face) for face in faces])
        if self.faces == 'cliques':
            for size in range(TRIANGLE_NODES + 1, self.dimension + 2):
                cells.append([tuple(index[node] for node in clique) for clique in self._cliques(size)])
        complex_ = xgi.SimplicialComplex()
        complex_.add_nodes_from(range(self.graph.number_of_nodes()))
        complex_.add_simplices_from([list(cell) for degree in cells[1:] for cell in degree])
        given = {frozenset(cell): self._parity(cell) for degree in cells[1:] for cell in degree}
        orientations = {
            identifier: given.get(frozenset(complex_.edges.members(identifier)), 0)
            for identifier in complex_.edges.filterby('order', 1, mode='geq')
        }
        return tuple(self._simplicial_incidence(complex_, orientations, cells, order) for order in range(1, len(cells)))

    def _simplicial_incidence(
        self,
        complex_: xgi.SimplicialComplex,
        orientations: dict[object, int],
        cells: list[list[tuple[int, ...]]],
        order: int,
    ) -> scipy.sparse.csr_array:
        """Return the incidence of an order of the boundary matrices of XGI.

        It is in the numbering of the cells.
        """
        lower = {frozenset(cell): number for number, cell in enumerate(cells[order - 1])}
        upper = {frozenset(cell): number for number, cell in enumerate(cells[order])}
        shape = (len(cells[order]), len(cells[order - 1]))
        if not all(shape):
            return self._sparse(np.empty(0, np.intp), np.empty(0, np.intp), np.empty(0), shape)
        boundary, rows, columns = xgi.linalg.hodge_matrix.boundary_matrix(complex_, order, orientations, index=True)
        lows = np.asarray(
            [
                lower[frozenset([rows[row]] if order == 1 else complex_.edges.members(rows[row]))]
                for row in range(boundary.shape[0])
            ],
            dtype=np.intp,
        )
        highs = np.asarray(
            [upper[frozenset(complex_.edges.members(columns[column]))] for column in range(boundary.shape[1])],
            dtype=np.intp,
        )
        found_rows, found_columns = np.nonzero(boundary)
        return self._sparse(highs[found_columns], lows[found_rows], boundary[found_rows, found_columns], shape)

    def _prepare_network(self, place: NetworkSpace) -> None:
        """Set the graph, the faces, and the incidence of the cells of a network."""
        self.graph = place.graph
        if any(first == second for first, second in nx.selfloop_edges(self.graph)):
            error_msg = 'A cell complex of a network holds no loops. Remove the loops of the graph.'
            raise SpaceError(error_msg)
        sources, targets = NetworkSpace._get_edges(self, self.graph)
        faces = self._network_faces(place)
        self._check_faces(faces)
        self.hyperedges = faces
        if self._is_simplicial(faces):
            self.incidences_ = self._simplicial_incidences(faces, sources, targets)
        else:
            self.incidences_ = (self._edge_incidence(sources, targets), self._cycle_incidence(faces, sources, targets))
        self.stars_ = (np.ones(self.incidences_[0].shape[1]), np.ones(sources.size), np.ones(len(faces)))
        self.boundaries_, self.boundary_signs_ = self._boundary_tables(faces, sources, targets)
        self.link_axes_ = np.empty(0, dtype=np.intp)
        self.face_planes_ = np.empty((0, 2), dtype=np.intp)

    def _boundary_tables(
        self,
        faces: list[tuple[object, ...]],
        sources: Indices,
        targets: Indices,
    ) -> tuple[Indices, Array]:
        """Return the edges around every face of a network and their signs, padded.

        The edges come in the cyclic order of the nodes of the face.
        """
        index = {node: number for number, node in enumerate(self.graph.nodes)}
        ends = zip(sources.tolist(), targets.tolist(), strict=True)
        edges = {(first, second): edge for edge, (first, second) in enumerate(ends)}
        cycles = [[index[node] for node in face] for face in faces]
        width = max((len(cycle) for cycle in cycles), default=0)
        table = np.full((len(cycles), width), -1, dtype=np.intp)
        signs = np.zeros((len(cycles), width))
        for row, nodes in enumerate(cycles):
            for column, (first, second) in enumerate(zip(nodes, (*nodes[1:], nodes[0]), strict=True)):
                forward = (first, second) in edges
                table[row, column] = edges[(first, second)] if forward else edges[(second, first)]
                signs[row, column] = 1.0 if forward else -1.0
        return table, signs

    def _coface_tables(self, matrix: scipy.sparse.csr_array) -> tuple[npt.NDArray[np.intp], Array]:
        """Return the cofaces of every cell of an incidence and their signs.

        The cofaces are padded with their number.
        """
        columns = matrix.T.tocsr()
        counts = np.diff(columns.indptr)
        table = np.full((columns.shape[0], int(counts.max(initial=0))), columns.shape[1], dtype=np.intp)
        signs = np.zeros(table.shape)
        rows = np.repeat(np.arange(columns.shape[0]), counts)
        slots = np.arange(columns.indices.size) - np.repeat(columns.indptr[:-1], counts)
        table[rows, slots] = columns.indices
        signs[rows, slots] = columns.data
        return table, signs

    def _get_edges(self, graph: nx.Graph) -> tuple[Indices, Indices]:
        """Return the source and target of every link or edge.

        A link runs along its axis on a lattice, and an edge is that of a network.
        """
        if isinstance(self.space, LatticeSpace):
            return self._lattice_links(self.space)[:2]
        return super()._get_edges(graph)

    def _check(self) -> None:
        """Raise an error for a space, faces, or a dimension the complex cannot take.

        Raises:
            SpaceError:
                The space is neither a square lattice nor a network, or a lattice is given faces.

            ValidationError:
                The dimension is below 2, or the faces are not known.
        """
        if not isinstance(self.space, LatticeSpace | NetworkSpace):
            error_msg = f'A cell complex derives from a lattice or a network space, not {self.space!r}.'
            raise SpaceError(error_msg)
        self.space._validated()
        if isinstance(self.space, LatticeSpace) and (self.space.geometry != 'square' or self.faces is not None):
            error_msg = 'A cell complex of a lattice takes a square lattice and no faces, which the lattice gives.'
            raise SpaceError(error_msg)
        if self.dimension < LOWEST_DIMENSION:
            error_msg = f'The dimension of a cell complex is {self.dimension!r}. Give 2 or more.'
            raise ValidationError(error_msg)
        if isinstance(self.faces, str) and self.faces not in FACES:
            error_msg = f'Unknown faces {self.faces!r}. Use None, one of {", ".join(FACES)}, or a sequence of cycles.'
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set the graph, the hyperedges, and the incidence of the cells, then the network.

        It then sets the attributes of the network.

        Args:
            context:
                Nothing, since a cell complex derives all it holds from its arguments.

        Raises:
            SpaceError:
                A face uses a pair of nodes that the graph does not join, or the network has a loop.
        """
        place = self.space.prepare()
        if isinstance(place, LatticeSpace):
            self._prepare_lattice(place)
        else:
            self._prepare_network(place)
        super()._prepare(context)
        tables = [self._coface_tables(matrix) for matrix in self.incidences_]
        self.cofaces_ = tuple(table for table, _ in tables)
        self.orientations_ = tuple(signs for _, signs in tables)
