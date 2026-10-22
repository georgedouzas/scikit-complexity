"""Derive the sparsity pattern of the Jacobian of a drift and its exact value from the compiled symbolic laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR

from ..base import Expression, ValidationError, symbols_of
from ._kernels import Kernel, Output, Reference, Source, member_indices
from ._printing import numpy_function
from ._reductions import CompiledReduction, ReductionPass
from ._volumes import FiniteVolume

type Vector = npt.NDArray[np.float64]
type Indices = npt.NDArray[np.intp]
type Sparse = scipy.sparse.csr_array
type Symmetry = tuple[Indices, tuple[int, ...], list[tuple[int, int]]]

PATTERN_LIMIT = 4_000_000
STENCIL = 2


class _TooDense(Exception):  # noqa: N818
    """Raised internally when the dense blocks of a pattern exceed the size limit."""


def _selection(
    rows: npt.NDArray[np.intp],
    columns: npt.NDArray[np.intp],
    values: Any,  # noqa: ANN401
    size: int,
    height: int | None = None,
) -> Sparse:
    """Return the sparse matrix with the given values at the given rows and columns, summing duplicates."""
    shape = (int(rows.max(initial=-1)) + 1 if height is None else height, size)
    data = np.broadcast_to(np.asarray(values, dtype=float), rows.shape)
    return scipy.sparse.csr_array((data, (rows, columns)), shape=shape)


def _boolean(matrix: Any) -> Sparse:  # noqa: ANN401
    """Return the nonzero structure of a sparse matrix as a boolean CSR matrix."""
    matrix = scipy.sparse.csr_array(matrix)
    matrix.eliminate_zeros()
    return scipy.sparse.csr_array((np.ones(matrix.nnz, dtype=bool), matrix.indices, matrix.indptr), shape=matrix.shape)


def _product(first: Any, second: Any) -> Sparse:  # noqa: ANN401
    """Return the structure of the product of two boolean structures."""
    return _boolean(scipy.sparse.csr_array(first, dtype=float) @ scipy.sparse.csr_array(second, dtype=float))


@dataclass
class _Partials:
    """Partial derivatives of expressions by the symbols they read, compiled into one NumPy function."""

    time: Expression
    arguments: list[Expression]
    expressions: list[Expression]
    by: list[Expression]
    _function: Callable[..., list[Any]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Differentiate every expression by every symbol it reads and compile the derivatives."""
        self.pairs = [
            (index, item)
            for index, expression in enumerate(self.expressions)
            for item in self.by
            if item in SR(expression).variables()
        ]
        derivatives = [SR(self.expressions[index]).diff(item) for index, item in self.pairs]
        self._function = numpy_function(self.time, self.arguments, derivatives) if derivatives else None

    def evaluate(self, t: float, values: list[Any]) -> list[tuple[int, Expression, Any]]:
        """Return every derivative with the index of its expression and the symbol it is taken by."""
        if self._function is None:
            return []
        results = self._function(t, *values)
        return [(index, item, value) for (index, item), value in zip(self.pairs, results, strict=True)]


def _output_symbols(output: Output) -> set[Expression] | None:
    """Return the symbols an output reads, or `None` for a callable law, which reads every state."""
    if output.expression is not None:
        return set(SR(output.expression).variables())
    if output.arguments is not None:
        return symbols_of(list(output.arguments))
    return None


def _neighbor_columns(reference: Reference) -> tuple[Indices, Indices]:
    """Return the members that read a neighbour and the flat index of the neighbour each reads."""
    read = reference.neighbor
    table = np.asarray(read.table, dtype=np.intp)
    entry = reference.entry
    members = np.arange(table.size, dtype=np.intp)
    inside = table < entry.rows
    columns = entry.start + table[inside] * entry.columns + reference.column
    return members[inside], columns.astype(np.intp)


def _mirror_projections(symmetries: Sequence[Symmetry], size: int) -> list[Sparse]:
    """Return the matrices that project the rates of symmetric states onto their odd or even parts, axis by axis."""
    projections = []
    for indices, dimensions, signs in symmetries:
        grid = np.arange(int(np.prod(dimensions))).reshape(dimensions)
        free = np.setdiff1d(np.arange(size), indices)
        for axis, sign in signs:
            mirror = np.roll(np.flip(grid, axis=axis), 1, axis=axis).ravel()
            rows = np.concatenate([free, indices, indices])
            columns = np.concatenate([free, indices, indices[mirror]])
            values = np.concatenate([np.ones(free.size), np.full(indices.size, 0.5), np.full(indices.size, sign / 2)])
            projections.append(scipy.sparse.csr_array((values, (rows, columns)), shape=(size, size)))
    return projections


def _volume_pattern(volume: FiniteVolume, size: int) -> Sparse:
    """Return the band of the reconstruction stencil of a finite-volume field, coupling all its conserved states."""
    grid = np.arange(volume.size).reshape(volume.dimensions)
    kinds = {kind for kind, _ in volume.boundaries}
    rows, columns = [], []
    for axis, length in enumerate(volume.dimensions):
        for offset in range(-STENCIL, STENCIL + 1):
            shifted = np.arange(length) + offset
            neighbours = []
            if 'periodic' in kinds:
                neighbours.append(np.mod(shifted, length))
            if kinds - {'periodic'}:
                neighbours.append(np.clip(shifted, 0, length - 1))
            for items in neighbours:
                rows.append(grid.ravel())
                columns.append(np.take(grid, items, axis=axis).ravel())
    cells, others = np.concatenate(rows), np.concatenate(columns)
    pairs = [(first + cells, second + others) for first in volume.starts for second in volume.starts]
    return _selection(
        np.concatenate([item for item, _ in pairs]),
        np.concatenate([item for _, item in pairs]),
        1.0,
        size,
        height=size,
    )


@dataclass
class ExactJacobian:
    """Jacobian of the drift of an ODE problem, derived from its symbolic laws, with its sparsity pattern.

    `pattern` states every entry the laws can make nonzero: member and shared reads, the neighbour tables of fields,
    the sparse matrices of finite element operators, the reconstruction stencils of finite volumes, the rank-one blocks
    of reductions, and dense blocks where couplings are known only at run time, such as pairs, spectral operators, and
    callables. Calling the Jacobian evaluates it exactly by differentiating the laws, with the chain rule through
    neighbour reads, finite element operators, and reductions.

    Every reading member depends on every member a reduction reads, so the Jacobian has the rank-one block
    `dF_i/dR dR/dx_j` of every reduction `R`, with `dR/dx_j` the derivative of the reduced member's argument, divided by
    the count for a mean, weighted by `W_ij` with weights, at the first extreme member for a maximum or a minimum, and 0
    for a count.

    Args:
        kernels:
            The kernels of the rate laws.

        reductions:
            The reductions the kernels read.

        size:
            The size of the state vector.

        time:
            The time symbol.

        volumes:
            The conservation laws of finite-volume fields.

        symmetries:
            The state indices, lattice shape, and axes with their signs of every symmetric field state.

        constrained:
            The fixed rows, whose rates are zero.
    """

    kernels: tuple[Kernel, ...]
    reductions: ReductionPass
    size: int
    time: Expression
    volumes: tuple[FiniteVolume, ...] = ()
    symmetries: tuple[Symmetry, ...] = ()
    constrained: Indices = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    _kernel_partials: list[_Partials] | None = field(default=None, init=False, repr=False)
    _reduction_partials: dict[Expression, _Partials] = field(default_factory=dict, init=False, repr=False)
    _operand_partials: dict[int, _Partials] = field(default_factory=dict, init=False, repr=False)
    _pattern: Sparse | None = field(default=None, init=False, repr=False)
    _patterned: bool = field(default=False, init=False, repr=False)

    def validate(self) -> None:
        """Raise an error when a law has no symbolic form whose exact Jacobian can be derived.

        Raises:
            ValidationError:
                A rate law holds a callable, a pair sum, or a spectral operator, or a field has conservation laws.
        """
        if self.reductions.neighbourhoods:
            error_msg = (
                'The rate laws reduce over neighbourhoods, whose exact Jacobian is not derived. Use the automatic '
                "option, jacobian='auto', which estimates it by finite differences."
            )
            raise ValidationError(error_msg)
        for volume in self.volumes:
            error_msg = (
                f'The conservation laws of {volume.owner} have no exact Jacobian. Use the automatic option, '
                "jacobian='auto', which estimates it by finite differences on its sparsity pattern."
            )
            raise ValidationError(error_msg)
        for kernel in self.kernels:
            references = [kernel.references[item] for item in kernel.arguments]
            spectral = any(
                reference.source is Source.DERIVED and reference.neighbor.matrix is None for reference in references
            )
            pairs = kernel.pairs is not None or any(
                reference.source in {Source.FIRST, Source.SECOND} for reference in references
            )
            if pairs or kernel.has_python_functions or spectral:
                error_msg = (
                    f'The rate laws of {kernel.owner} hold callables, pairs, or spectral operators, whose exact '
                    "Jacobian is not derived. Use the automatic option, jacobian='auto', which estimates it by finite "
                    'differences on its sparsity pattern.'
                )
                raise ValidationError(error_msg)

    def _compile(self) -> list[_Partials]:
        """Return the partial derivatives of every kernel, compiling them and those of the reductions once."""
        if self._kernel_partials is not None:
            return self._kernel_partials
        self.validate()
        kernel_partials = []
        for kernel in self.kernels:
            expressions = [output.expression for output in kernel.outputs]
            by = [item for item in kernel.arguments if _is_variable(kernel.references[item])]
            kernel_partials.append(_Partials(self.time, kernel.arguments, expressions, by))
        for reduction in self.reductions.reductions:
            by = [item for item in reduction.symbols if reduction.reads[item].is_state or item in self._by_symbol]
            self._reduction_partials[reduction.symbol] = _Partials(
                self.time,
                reduction.symbols,
                [reduction.argument],
                by,
            )
        self._kernel_partials = kernel_partials
        return kernel_partials

    @property
    def _by_symbol(self) -> dict[Expression, CompiledReduction]:
        """Reductions by placeholder symbol."""
        return {item.symbol: item for item in self.reductions.reductions}

    def _member_gradients(
        self,
        reduction: CompiledReduction,
        t: float,
        x: Vector,
        q: Vector,
        gradients: dict[Expression, Sparse],
    ) -> Sparse:
        """Return the gradient of every reduced member's argument by the state, one row per member."""
        total = scipy.sparse.csr_array((reduction.size, self.size))
        members = np.arange(reduction.size)
        for _, item, value in self._reduction_partials[reduction.symbol].evaluate(t, reduction.gather(x, q)):
            derivative = np.broadcast_to(np.asarray(value, dtype=float), (reduction.size,))
            read = reduction.reads[item]
            if item in gradients:
                nested = gradients[item]
                block = nested if nested.shape[0] == reduction.size else scipy.sparse.vstack([nested] * reduction.size)
                total = total + scipy.sparse.diags_array(derivative) @ block
                continue
            columns = np.broadcast_to(np.asarray(read.index, dtype=np.intp), (reduction.size,))
            total = total + _selection(members, columns, derivative, self.size)
        return scipy.sparse.csr_array(total)

    def _reduction_gradient(
        self,
        reduction: CompiledReduction,
        t: float,
        x: Vector,
        q: Vector,
        gradients: dict[Expression, Sparse],
    ) -> Sparse:
        """Return the gradient of a reduction by the state, one row per reading member for weights."""
        argument, condition = reduction.values(t, x, q)
        mask = condition.astype(float)
        selected = float(mask.sum())
        members = self._member_gradients(reduction, t, x, q, gradients)
        if reduction.kind == 'count' or not selected:
            return scipy.sparse.csr_array((1 if reduction.weights is None else reduction.weights.shape[0], self.size))
        if reduction.weights is not None:
            weighted = reduction.weights @ scipy.sparse.diags_array(mask) @ members
            return scipy.sparse.csr_array(weighted / selected if reduction.kind == 'mean' else weighted)
        if reduction.kind in {'max', 'min'}:
            values = np.where(condition, argument, -np.inf if reduction.kind == 'max' else np.inf)
            first = int(np.argmax(values) if reduction.kind == 'max' else np.argmin(values))
            return scipy.sparse.csr_array(members[[first]])
        if reduction.kind == 'var':
            mean = float(argument[condition].mean())
            factors = 2 * mask * (argument - mean) / selected
        else:
            factors = mask / selected if reduction.kind == 'mean' else mask
        return scipy.sparse.csr_array(factors[np.newaxis, :] @ members)

    def _reduced_rows(self, kernel: Kernel, reference: Reference, rows: Sparse) -> Sparse:
        """Return the rows of a reduction's gradient or support read by every evaluation index of a kernel."""
        if reference.entry.members is None:
            rows = rows[[min(reference.column, rows.shape[0] - 1)]]
        return rows if rows.shape[0] == kernel.rows else scipy.sparse.vstack([rows] * kernel.rows, format='csr')

    def _symbol_gradient(
        self,
        kernel: Kernel,
        item: Expression,
        *,
        t: float,
        x: Vector,
        q: Vector,
        gradients: dict[Expression, Sparse],
    ) -> Sparse | None:
        """Return the gradient of a symbol by the state, one row per evaluation index, or `None` for parameters."""
        reference = kernel.references[item]
        rows = np.arange(kernel.rows)
        if reference.source is Source.REDUCED:
            return self._reduced_rows(kernel, reference, gradients[item])
        if reference.source is Source.DERIVED:
            read = reference.neighbor
            key = id(read)
            if key not in self._operand_partials:
                by = [symbol for symbol in read.symbols if _is_variable(kernel.references[symbol])]
                self._operand_partials[key] = _Partials(self.time, read.symbols, [read.operand], by)
            values = [kernel.read(symbol, t, x, q) for symbol in read.symbols]
            total = scipy.sparse.csr_array((kernel.rows, self.size))
            for _, symbol, value in self._operand_partials[key].evaluate(t, values):
                inner = self._symbol_gradient(kernel, symbol, t=t, x=x, q=q, gradients=gradients)
                if inner is not None:
                    derivative = np.broadcast_to(np.asarray(value, dtype=float), (kernel.rows,))
                    total = total + scipy.sparse.diags_array(derivative) @ inner
            return scipy.sparse.csr_array(read.matrix @ total)
        if not reference.is_state:
            return None
        if reference.source is Source.NEIGHBOR:
            members, columns = _neighbor_columns(reference)
            return _selection(members, columns, 1.0, self.size, height=kernel.rows)
        return _selection(rows, member_indices(reference, kernel.rows), 1.0, self.size, height=kernel.rows)

    def __call__(self, t: float, x: Vector, p: Vector) -> Sparse:
        """Return the Jacobian of the drift at a state as a sparse matrix.

        Args:
            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            jacobian:
                The matrix of `dF_i/dx_j`.

        Raises:
            ValidationError:
                A rate law holds a callable, a pair sum, or a spectral operator, or a field has conservation laws.
        """
        partials = self._compile()
        q = self.reductions.extend(t, x, p)
        gradients: dict[Expression, Sparse] = {}
        for reduction in self.reductions.reductions:
            gradients[reduction.symbol] = self._reduction_gradient(reduction, t, x, q, gradients)
        jacobian = scipy.sparse.csr_array((self.size, self.size))
        for kernel, kernel_partials in zip(self.kernels, partials, strict=True):
            for index, item, value in kernel_partials.evaluate(t, kernel.gather(t, x, q)):
                gradient = self._symbol_gradient(kernel, item, t=t, x=x, q=q, gradients=gradients)
                if gradient is None:
                    continue
                target = member_indices(kernel.outputs[index].target, kernel.rows)  # type: ignore[arg-type]
                scatter = _selection(np.arange(kernel.rows), target, 1.0, self.size).T
                derivative = np.broadcast_to(np.asarray(value, dtype=float), (kernel.rows,))
                jacobian = jacobian + scatter @ scipy.sparse.diags_array(derivative) @ gradient
        for projection in _mirror_projections(self.symmetries, self.size):
            jacobian = projection @ jacobian
        return scipy.sparse.csr_array(self._free_rows() @ jacobian)

    def _free_rows(self) -> Sparse:
        """Return the diagonal matrix that empties the fixed rows."""
        diagonal = np.ones(self.size)
        diagonal[self.constrained] = 0.0
        return scipy.sparse.diags_array(diagonal, format='csr')

    def pattern(self) -> Sparse | None:
        """Return the boolean structure of the Jacobian, or `None` when its dense blocks are too large to state.

        Returns:
            pattern:
                A boolean sparse matrix that holds every entry the Jacobian can make nonzero and the diagonal of free
                rows, with fixed rows empty, or `None` when the dense blocks exceed `PATTERN_LIMIT` entries or the
                laws reduce over neighbourhoods, whose neighbours can change.
        """
        if not self._patterned:
            try:
                self._pattern = self._build_pattern()
            except _TooDense:
                self._pattern = None
            self._patterned = True
        return self._pattern

    def _build_pattern(self) -> Sparse:
        """Return the boolean structure of the Jacobian, raising `_TooDense` above the size limit."""
        self._dense = 0
        if self.reductions.neighbourhoods:
            raise _TooDense
        supports: dict[Expression, Sparse] = {}
        for reduction in self.reductions.reductions:
            supports[reduction.symbol] = self._reduction_support(reduction, supports)
        blocks = [scipy.sparse.eye_array(self.size, format='csr')]
        for kernel in self.kernels:
            blocks.extend(self._kernel_pattern(kernel, supports))
        blocks.extend(_volume_pattern(volume, self.size) for volume in self.volumes)
        total = scipy.sparse.csr_array((self.size, self.size))
        for block in blocks:
            total = total + scipy.sparse.csr_array(block, dtype=float)
        pattern = _boolean(total)
        for projection in _mirror_projections(self.symmetries, self.size):
            pattern = _product(_boolean(projection), pattern)
        return _product(self._free_rows(), pattern)

    def _note_dense(self, entries: int) -> None:
        """Count the entries of a dense block and stop above the size limit."""
        self._dense += entries
        if self._dense > PATTERN_LIMIT:
            raise _TooDense

    def _dense_rows(self, rows: int, columns: Indices) -> Sparse:
        """Return the block where every evaluation index reads every one of the columns."""
        columns = np.unique(columns)
        self._note_dense(rows * columns.size)
        return _selection(np.repeat(np.arange(rows), columns.size), np.tile(columns, rows), 1.0, self.size, rows)

    def _all_columns(self, kernel: Kernel, item: Expression, supports: dict[Expression, Sparse]) -> Indices:
        """Return every state index a symbol can read at any evaluation index."""
        reference = kernel.references[item]
        if reference.source is Source.REDUCED:
            return np.unique(supports[item].indices).astype(np.intp)
        if reference.source is Source.DERIVED:
            parts = [self._all_columns(kernel, symbol, supports) for symbol in reference.neighbor.symbols]
            return np.unique(np.concatenate([np.zeros(0, dtype=np.intp), *parts]))
        if not reference.is_state:
            return np.zeros(0, dtype=np.intp)
        if reference.source is Source.SINGLE:
            return np.array([reference.index], dtype=np.intp)
        return reference.flat_indices.astype(np.intp)

    def _symbol_support(self, kernel: Kernel, item: Expression, supports: dict[Expression, Sparse]) -> Sparse | None:
        """Return the states a symbol reads at every evaluation index of a member kernel, or `None` for parameters."""
        reference = kernel.references.get(item)
        if reference is None:
            return None
        if reference.source is Source.REDUCED:
            block = self._reduced_rows(kernel, reference, supports[item])
            self._note_dense(block.nnz)
            return block
        if reference.source is Source.DERIVED:
            return self._derived_support(kernel, item, supports)
        if not reference.is_state:
            return None
        return self._state_support(kernel, reference)

    def _derived_support(self, kernel: Kernel, item: Expression, supports: dict[Expression, Sparse]) -> Sparse:
        """Return the states a derived read reads, through its sparse operator or as a dense block."""
        read = kernel.references[item].neighbor
        if read.matrix is None:
            return self._dense_rows(kernel.rows, self._all_columns(kernel, item, supports))
        operand = scipy.sparse.csr_array((kernel.rows, self.size))
        for symbol in read.symbols:
            block = self._symbol_support(kernel, symbol, supports)
            if block is not None:
                operand = operand + scipy.sparse.csr_array(block, dtype=float)
        return _product(_boolean(read.matrix), operand)

    def _state_support(self, kernel: Kernel, reference: Reference) -> Sparse:
        """Return the states a state symbol reads at every evaluation index."""
        rows = kernel.rows
        if reference.source is Source.NEIGHBOR:
            members, columns = _neighbor_columns(reference)
            return _selection(members, columns, 1.0, self.size, height=rows)
        if reference.source is Source.SINGLE:
            return _selection(np.arange(rows), np.full(rows, reference.index, dtype=np.intp), 1.0, self.size, rows)
        if reference.source is Source.MEMBER and reference.entry.rows == rows:
            return _selection(np.arange(rows), reference.flat_indices, 1.0, self.size, height=rows)
        return self._dense_rows(rows, reference.flat_indices)

    def _kernel_pattern(self, kernel: Kernel, supports: dict[Expression, Sparse]) -> list[Sparse]:
        """Return the blocks of the pattern that the outputs of one kernel add."""
        blocks = []
        everything = np.arange(self.size, dtype=np.intp)
        for output in kernel.outputs:
            target: Reference = output.target  # type: ignore[assignment]
            used = _output_symbols(output)
            if kernel.pairs is not None or target.source in {Source.FIRST, Source.SECOND}:
                targets = np.unique(target.flat_indices if target.source is not Source.SINGLE else [target.index])
                columns = (
                    everything
                    if used is None
                    else np.unique(
                        np.concatenate(
                            [np.zeros(0, dtype=np.intp)]
                            + [self._all_columns(kernel, item, supports) for item in used if item in kernel.references],
                        ),
                    )
                )
                block = self._dense_rows(targets.size, columns)
                blocks.append(_selection(targets, np.arange(targets.size), 1.0, targets.size, self.size) @ block)
                continue
            scatter = _selection(np.arange(kernel.rows), member_indices(target, kernel.rows), 1.0, self.size).T
            if used is None:
                blocks.append(scatter @ self._dense_rows(kernel.rows, everything))
                continue
            for item in used:
                support = self._symbol_support(kernel, item, supports)
                if support is not None:
                    blocks.append(scatter @ support)
        return blocks

    def _reduction_support(self, reduction: CompiledReduction, supports: dict[Expression, Sparse]) -> Sparse:
        """Return the states a reduction depends on, one row per reading member for weights."""
        members = np.arange(reduction.size)
        support = scipy.sparse.csr_array((reduction.size, self.size), dtype=bool)
        for item in reduction.symbols:
            read = reduction.reads[item]
            if item in supports:
                nested = supports[item]
                block = nested if nested.shape[0] == reduction.size else scipy.sparse.vstack([nested] * reduction.size)
                support = support + (block != 0)
            elif read.is_state:
                columns = np.broadcast_to(np.asarray(read.index, dtype=np.intp), (reduction.size,))
                support = support + (_selection(members, columns, 1.0, self.size) != 0)
        if reduction.kind == 'count':
            return scipy.sparse.csr_array((1, self.size), dtype=bool)
        if reduction.weights is not None:
            return scipy.sparse.csr_array((abs(reduction.weights) != 0).astype(float) @ support.astype(float) != 0)
        return scipy.sparse.csr_array(np.asarray(support.astype(float).sum(axis=0)).reshape(1, -1) != 0)


def _is_variable(reference: Reference) -> bool:
    """Check whether a reference stands for a state, a reduction, or a derived read, so derivatives reach the state."""
    return reference.source in {Source.REDUCED, Source.DERIVED} or reference.is_state
