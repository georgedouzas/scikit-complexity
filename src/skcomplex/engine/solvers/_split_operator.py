"""Linear part of the rates of a problem, in Fourier blocks for spectral fields and a sparse matrix otherwise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import splu

from ...core.errors import SolverError
from ...core.spaces import (
    LatticeNeighborhood,
    LatticeSpace,
    MeshSpace,
    NetworkSpace,
)
from ...core.types import Vector
from ..compile._field_lowering import FieldLowering
from ..problems import LATTICE_BOUNDARIES, ExpressionPrinter, Problem, symbol

TIME = symbol('t')


type Complex = npt.NDArray[np.complex128]


@dataclass
class _SpectralBlock:
    """Linear terms of the states of one spectral field, a small matrix per Fourier mode."""

    dimensions: tuple[int, ...]
    rows: list[npt.NDArray[np.intp]]
    matrix: Complex

    def transform(self, x: Vector) -> Complex:
        """Return the Fourier modes of every state of the block, stacked on the last axis."""
        return np.stack([np.fft.rfftn(np.reshape(x[rows], self.dimensions)) for rows in self.rows], axis=-1)

    def inverse(self, modes: Complex, out: Vector) -> None:
        """Write the states of Fourier modes stacked on the last axis."""
        for index, rows in enumerate(self.rows):
            out[rows] = np.fft.irfftn(modes[..., index], s=self.dimensions, axes=range(len(self.dimensions))).ravel()

    def apply(self, operator: Complex, x: Vector, out: Vector) -> None:
        """Write a matrix of every mode applied to the block's states."""
        self.inverse(np.einsum('...ij,...j->...i', operator, self.transform(x)), out)


@dataclass
class SplitOperator:
    """Linear part of the rates of a problem, in Fourier blocks for spectral fields and a sparse matrix otherwise.

    Args:
        size:
            The size of the state vector.

        blocks:
            The spectral blocks.

        sparse:
            The sparse matrix of the other linear terms, over the whole state vector.

        covered:
            The state indices of the spectral blocks.
    """

    size: int
    blocks: list[_SpectralBlock] = field(default_factory=list)
    sparse: csr_matrix | None = None
    covered: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    _solvers: dict[float, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, problem: Problem, p: Vector) -> 'SplitOperator':
        """Return the linear part of a problem's rates at parameter values.

        Args:
            problem:
                The compiled problem.

            p:
                The parameter vector.

        Returns:
            operator:
                The linear part.
        """
        operator = cls(problem.states.size)
        groups: dict[int, list[Any]] = {}
        triplets: list[tuple[Any, Any, csr_matrix]] = []
        for term in problem.linear:
            coefficient = cls._coefficient(term, p)
            if term.place.discretization != 'spectral':
                triplets.append((term.target, term.source, cls._term_matrix(term, coefficient)))
            elif not np.ndim(coefficient) or np.ptp(coefficient) == 0:
                groups.setdefault(id(term.place), []).append((term, complex(np.ravel(coefficient)[0])))
        operator.sparse = cls._sparse(triplets, problem)
        operator.blocks = [cls._spectral_block(terms) for terms in groups.values()]
        rows = [row for block in operator.blocks for row in block.rows]
        operator.covered = np.concatenate([np.zeros(0, dtype=np.intp), *rows]).astype(np.intp)
        return operator

    def apply(self, x: Vector) -> Vector:
        """Return the linear part applied to a state vector.

        Args:
            x:
                The state vector.

        Returns:
            values:
                The linear part of the rates.
        """
        out = np.zeros(self.size)
        if self.sparse is not None:
            out += self.sparse @ x
        for block in self.blocks:
            block.apply(block.matrix, x, out)
        return out

    def solve(self, factor: float, rhs: Vector) -> Vector:
        """Return the solution `y` of `(I - factor L) y = rhs`.

        Args:
            factor:
                The factor of the linear part, such as the step for implicit Euler.

            rhs:
                The right-hand side.

        Returns:
            solution:
                The solution.
        """
        if factor not in self._solvers:
            sparse = None
            if self.sparse is not None:
                sparse = splu((identity(self.size, format='csc') - factor * self.sparse).tocsc())
            inverses = [np.linalg.inv(np.eye(block.matrix.shape[-1]) - factor * block.matrix) for block in self.blocks]
            self._solvers[factor] = (sparse, inverses)
        sparse, inverses = self._solvers[factor]
        out = sparse.solve(rhs) if sparse is not None else rhs.copy()
        for block, inverse in zip(self.blocks, inverses, strict=True):
            block.apply(inverse, rhs, out)
        return out

    @staticmethod
    def _lattice_operator(name: str, axis: int, place: Any, reference: Any) -> csr_matrix:  # noqa: ANN401
        """Return the stencil matrix of a finite-difference operator on a field state, without its boundary values."""
        space = place.space
        size = place.size_
        kind, _ = place._get_boundary(reference.entry.name.rpartition('.')[2])

        def shift(offset: Any) -> csr_matrix:  # noqa: ANN401
            lattice = LatticeSpace(space.shape_, boundary=LATTICE_BOUNDARIES[kind]).check().prepare()
            table = lattice._index_table(LatticeNeighborhood((offset,)))[:, 0]
            inside = table < size
            return csr_matrix((np.ones(inside.sum()), (np.flatnonzero(inside), table[inside])), shape=(size, size))

        if isinstance(space, NetworkSpace):
            adjacency = csr_matrix(nx.to_scipy_sparse_array(space.graph, nodelist=list(space.graph.nodes), weight=None))
            if name != 'laplacian':
                error_msg = f'Only Laplacians act on fields on networks, but a term takes the {name}.'
                raise SolverError(error_msg)
            return adjacency - diags(np.asarray(adjacency.sum(axis=1)).ravel())
        units = [tuple(int(other == index) for other in range(space.n_dim_)) for index in range(space.n_dim_)]
        spacings = space.spacings_
        if name == 'laplacian':
            total = csr_matrix((size, size))
            for unit, step in zip(units, spacings, strict=True):
                total = total + (shift(unit) + shift(tuple(-item for item in unit)) - 2 * identity(size)) / step**2
            return total
        if name == 'partial':
            unit = units[axis]
            return (shift(unit) - shift(tuple(-item for item in unit))) / (2 * spacings[axis])
        error_msg = f'The {name} of a finite-difference field has no stencil. Use a spectral field.'
        raise SolverError(error_msg)

    @staticmethod
    def _term_matrix(term: Any, coefficient: Any) -> csr_matrix:  # noqa: ANN401
        """Return the sparse matrix of a linear term of a finite-difference or finite element field."""
        place = term.place
        matrix = identity(place.size_, format='csr')
        for name, axis in reversed(term.chain):
            operator = (
                SplitOperator._mesh_operator(place.space, name)
                if isinstance(place.space, MeshSpace)
                else SplitOperator._lattice_operator(name, axis, place, term.source)
            )
            matrix = operator @ matrix
        return diags(np.broadcast_to(np.asarray(coefficient, dtype=float), place.size_)) @ matrix

    @staticmethod
    def _sparse(triplets: list[tuple[Any, Any, csr_matrix]], problem: Problem) -> csr_matrix | None:
        """Return the sparse matrix of linear terms over the whole state vector, without rows of fixed nodes."""
        if not triplets:
            return None
        size = problem.states.size
        rows, columns, values = [], [], []
        for target, source, matrix in triplets:
            block = matrix.tocoo()
            rows.append(SplitOperator._rows(target)[block.row])
            columns.append(SplitOperator._rows(source)[block.col])
            values.append(block.data)
        sparse = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        )
        if problem.constrained.size:
            keep = np.ones(size)
            keep[problem.constrained] = 0.0
            sparse = diags(keep) @ sparse
        return sparse

    @staticmethod
    def _spectral_block(terms: list[tuple[Any, complex]]) -> '_SpectralBlock':
        """Return the Fourier block of the linear terms of the states of one spectral field."""
        place = terms[0][0].place
        states: dict[tuple[str, int], Any] = {}
        for term, _ in terms:
            for reference in (term.target, term.source):
                states.setdefault((reference.entry.name, reference.column), reference)
        order = {key: index for index, key in enumerate(states)}
        dimensions, spacings = place.space.shape_, place.space.spacings_
        matrix = np.zeros((*np.fft.rfftn(np.zeros(dimensions)).shape, len(states), len(states)), dtype=complex)
        for term, coefficient in terms:
            multiplier: Any = 1.0
            for name, axis in term.chain:
                multiplier = multiplier * FieldLowering._spectral_multiplier(name, axis, dimensions, spacings)
            target = order[(term.target.entry.name, term.target.column)]
            source = order[(term.source.entry.name, term.source.column)]
            matrix[..., target, source] += coefficient * multiplier
        return _SpectralBlock(dimensions, [SplitOperator._rows(reference) for reference in states.values()], matrix)

    @staticmethod
    def _mesh_operator(space: MeshSpace, name: str) -> csr_matrix:
        """Return the finite element Laplacian of a mesh."""
        if name != 'laplacian':
            error_msg = f'Only Laplacians act on finite element fields, but a term takes the {name}.'
            raise SolverError(error_msg)
        return FieldLowering._mesh_laplacian(space)

    @staticmethod
    def _coefficient(term: Any, p: Vector) -> Any:  # noqa: ANN401
        """Return the value of a linear term's coefficient, a number or one value per member."""
        symbols = sorted(term.references, key=str)
        values = []
        for item in symbols:
            reference = term.references[item]
            column = reference.entry.view(p)[:, reference.column]
            values.append(float(column[0]) if reference.entry.members is None else column)
        printer = ExpressionPrinter(time=TIME, compiled=True).check().prepare(symbols)
        return printer.run([term.coefficient])(0.0, *values)[0]

    @staticmethod
    def _rows(reference: Any) -> npt.NDArray[np.intp]:  # noqa: ANN401
        """Return the flat state index of a component for every member."""
        return reference.flat_indices
