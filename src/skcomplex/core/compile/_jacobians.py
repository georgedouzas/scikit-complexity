"""Derive the exact Jacobian of a drift and its sparsity pattern by the chain rule through every reduction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR

from ..base import Expression, ValidationError
from ._kernels import Kernel, Reference, Source, member_indices
from ._printing import numpy_function
from ._reductions import CompiledReduction, ReductionPass

type Vector = npt.NDArray[np.float64]
type Sparse = scipy.sparse.csr_array

PATTERN_LIMIT = 4_000_000
SUPPORTED = frozenset({Source.MEMBER, Source.SINGLE, Source.REDUCED})


def _selection(
    rows: npt.NDArray[np.intp],
    columns: npt.NDArray[np.intp],
    values: Any,  # noqa: ANN401
    size: int,
) -> Sparse:
    """Return the sparse matrix with the given values at the given rows and columns, summing duplicates."""
    shape = (int(rows.max(initial=-1)) + 1, size)
    data = np.broadcast_to(np.asarray(values, dtype=float), rows.shape)
    return scipy.sparse.csr_array((data, (rows, columns)), shape=shape)


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


@dataclass
class ExactJacobian:
    """Jacobian of the drift of an ODE problem, exact through the reductions its laws read.

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
    """

    kernels: tuple[Kernel, ...]
    reductions: ReductionPass
    size: int
    time: Expression
    _kernel_partials: list[_Partials] | None = field(default=None, init=False, repr=False)
    _reduction_partials: dict[Expression, _Partials] = field(default_factory=dict, init=False, repr=False)

    def _check(self) -> None:
        """Raise an error for a law whose exact Jacobian cannot be derived."""
        for kernel in self.kernels:
            sources = {reference.source for reference in kernel.references.values()}
            if kernel.pairs is not None or kernel.has_python_functions or sources - SUPPORTED:
                error_msg = (
                    f'The rate laws of {kernel.owner} hold callables, pairs, or neighbour and operator reads, whose '
                    'exact Jacobian is not derived. Let the solver estimate the Jacobian by finite differences.'
                )
                raise ValidationError(error_msg)

    def _compile(self) -> list[_Partials]:
        """Return the partial derivatives of every kernel, compiling them and those of the reductions once."""
        if self._kernel_partials is not None:
            return self._kernel_partials
        self._check()
        self._kernel_partials = []
        for kernel in self.kernels:
            expressions = [output.expression for output in kernel.outputs]
            by = [item for item in kernel.arguments if _is_variable(kernel.references[item])]
            self._kernel_partials.append(_Partials(self.time, kernel.arguments, expressions, by))
        for reduction in self.reductions.reductions:
            by = [item for item in reduction.symbols if reduction.reads[item].is_state or item in self._by_symbol]
            self._reduction_partials[reduction.symbol] = _Partials(
                self.time,
                reduction.symbols,
                [reduction.argument],
                by,
            )
        return self._kernel_partials

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

    def _gradient(
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
                A rate law holds a callable, a pair sum, or a neighbour or operator read.
        """
        partials = self._compile()
        q = self.reductions.extend(t, x, p)
        gradients: dict[Expression, Sparse] = {}
        for reduction in self.reductions.reductions:
            gradients[reduction.symbol] = self._gradient(reduction, t, x, q, gradients)
        jacobian = scipy.sparse.csr_array((self.size, self.size))
        for kernel, kernel_partials in zip(self.kernels, partials, strict=True):
            for index, item, value in kernel_partials.evaluate(t, kernel.gather(t, x, q)):
                target = member_indices(kernel.outputs[index].target, kernel.rows)  # type: ignore[arg-type]
                jacobian = jacobian + self._block(kernel, target, item, value, gradients)
        return jacobian

    def _block(
        self,
        kernel: Kernel,
        target: npt.NDArray[np.intp],
        item: Expression,
        value: Any,  # noqa: ANN401
        gradients: dict[Expression, Sparse],
    ) -> Sparse:
        """Return the contribution of one partial derivative of a kernel output to the Jacobian."""
        reference = kernel.references[item]
        derivative = np.broadcast_to(np.asarray(value, dtype=float), (kernel.rows,))
        scatter = _selection(np.arange(kernel.rows), target, 1.0, self.size).T
        if reference.source is Source.REDUCED:
            gradient = gradients[item]
            if reference.entry.members is None:
                rows = gradient[[min(reference.column, gradient.shape[0] - 1)]]
                rows = scipy.sparse.vstack([rows] * kernel.rows)
            else:
                rows = gradient
            return scipy.sparse.csr_array(scatter @ scipy.sparse.diags_array(derivative) @ rows)
        columns = member_indices(reference, kernel.rows)
        return scipy.sparse.csr_array(scatter @ _selection(np.arange(kernel.rows), columns, derivative, self.size))

    def pattern(self) -> Sparse | None:
        """Return the boolean structure of the Jacobian, or `None` when a dense block of reductions is too large.

        Returns:
            pattern:
                A boolean sparse matrix that holds every entry the Jacobian can make nonzero and the diagonal, or
                `None` when the dense blocks of the reductions exceed `PATTERN_LIMIT` entries.

        Raises:
            ValidationError:
                A rate law holds a callable, a pair sum, or a neighbour or operator read.
        """
        self._compile()
        supports: dict[Expression, Sparse] = {}
        for reduction in self.reductions.reductions:
            supports[reduction.symbol] = self._support(reduction, supports)
        dense = 0
        pattern = scipy.sparse.eye_array(self.size, format='csr', dtype=bool)
        for kernel in self.kernels:
            for output in kernel.outputs:
                target = member_indices(output.target, kernel.rows)  # type: ignore[arg-type]
                scatter = _selection(np.arange(kernel.rows), target, 1.0, self.size).T
                for item in SR(output.expression).variables():
                    reference = kernel.references.get(item)
                    if reference is None or not _is_variable(reference):
                        continue
                    if reference.source is Source.REDUCED:
                        support = supports[item]
                        rows = (
                            support
                            if reference.entry.members is not None
                            else support[[min(reference.column, support.shape[0] - 1)]]
                        )
                        rows = rows if rows.shape[0] == kernel.rows else scipy.sparse.vstack([rows] * kernel.rows)
                        dense += rows.nnz
                        block = scatter @ rows
                    else:
                        block = scatter @ _selection(
                            np.arange(kernel.rows),
                            member_indices(reference, kernel.rows),
                            1.0,
                            self.size,
                        )
                    pattern = pattern + (block != 0)
        if dense > PATTERN_LIMIT:
            return None
        return scipy.sparse.csr_array(pattern, dtype=bool)

    def _support(self, reduction: CompiledReduction, supports: dict[Expression, Sparse]) -> Sparse:
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
    """Check whether a reference stands for a state or a reduction, so derivatives by it reach the state."""
    return reference.source is Source.REDUCED or (reference.is_state and reference.source in SUPPORTED)
