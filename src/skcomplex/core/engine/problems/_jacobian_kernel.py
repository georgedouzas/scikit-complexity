"""Derive the sparsity pattern of the Jacobian of a drift and its exact value from the compiled symbolic laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR, Expression
from scipy.special import logsumexp

from ...errors import ValidationError
from ...specs import ReadSpec, RunStateSpec
from ...types import Array, Indices
from .._utils import symbols_of
from ..types import ReadKind, Sparse
from ._base import BaseKernel
from ._expression_printer import ExpressionPrinter
from ._finite_volume_kernel import FiniteVolumeKernel
from ._law_kernel import LawKernel
from ._output_spec import OutputSpec
from ._partials_spec import PartialsSpec
from ._reduction_kernel import ReductionKernel
from ._reduction_schedule_spec import ReductionScheduleSpec
from ._reference_spec import ReferenceSpec
from ._symmetry_spec import SymmetrySpec
from ._too_dense import _TooDenseError

"""Index of the expression and symbol of every nonzero derivative, with the NumPy function of the derivatives."""

_PATTERN_LIMIT = 4_000_000


class JacobianKernel(BaseKernel):
    """Jacobian of the drift of an ODE problem, derived from its symbolic laws, with its sparsity pattern.

    `pattern` states every entry the laws can make nonzero: member and shared reads, the neighbour tables of fields,
    the sparse matrices of finite element operators, the reconstruction stencils of finite volumes, the rank-one blocks
    of reductions, and dense blocks where couplings are known only at run time, such as pairs, spectral operators, and
    callables. `prepare(problem)` checks that every law has a symbolic form and compiles its derivatives, and
    `run(state)` evaluates the Jacobian exactly into `values_` by differentiating the laws, with the chain rule through
    neighbour reads, finite element operators, and reductions.

    Every reading member depends on every member a reduction reads, so the Jacobian has the rank-one block
    `dF_i/dR dR/dx_j` of every reduction `R`, with `dR/dx_j` the derivative of the reduced member's argument, divided by
    the count for a mean, weighted by `W_ij` with weights, at the first extreme member for a maximum or a minimum, by
    the softmax of the arguments for a log-sum-exp, and 0 for a count.

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
            The conservation laws, such as those of finite-volume fields.

        symmetries:
            The mirror symmetries of the field states.

        constrained:
            The fixed rows, whose rates are zero.

    Attributes:
        values_:
            The Jacobian of the drift at the state of the last run, the sparse matrix of `dF_i/dx_j`.
    """

    values_: Sparse

    def __init__(
        self,
        kernels: tuple[LawKernel, ...],
        reductions: ReductionScheduleSpec,
        size: int,
        time: Expression,
        volumes: tuple[FiniteVolumeKernel, ...] = (),
        *,
        symmetries: tuple[SymmetrySpec, ...] = (),
        constrained: Indices | None = None,
    ) -> None:
        self.kernels = kernels
        self.reductions = reductions
        self.size = size
        self.time = time
        self.volumes = volumes
        self.symmetries = symmetries
        self.constrained = constrained

    def _check_exact_forms(self) -> None:
        """Raise an error when a law has no symbolic form whose exact Jacobian can be derived."""
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
                reference.source is ReadKind.DERIVED and reference.derived.matrix is None for reference in references
            )
            pairs = kernel.pairs is not None or any(
                reference.source in {ReadKind.FIRST, ReadKind.SECOND} for reference in references
            )
            if pairs or kernel.has_python_functions or spectral:
                error_msg = (
                    f'The rate laws of {kernel.owner} hold callables, pairs, or spectral operators, whose exact '
                    "Jacobian is not derived. Use the automatic option, jacobian='auto', which estimates it by finite "
                    'differences on its sparsity pattern.'
                )
                raise ValidationError(error_msg)

    def _prepare(self, context: object) -> None:
        """Check that every rate law has an exact Jacobian and compile the partial derivatives of the laws.

        Args:
            context:
                The problem the kernel serves, which it does not read.

        Raises:
            ValidationError:
                A rate law holds a callable, a pair sum, or a spectral operator, or a field has conservation laws.
        """
        del context
        self._check_exact_forms()
        self.kernel_partials_ = []
        for kernel in self.kernels:
            expressions = [output.expression for output in kernel.outputs]
            by = [item for item in kernel.arguments if self._is_variable(kernel.references[item])]
            self.kernel_partials_.append(self._partials(kernel.arguments, expressions, by))
        self.reduction_partials_ = {}
        for reduction in self.reductions.reductions:
            by = [item for item in reduction.symbols if reduction.reads[item].is_state or item in self._by_symbol]
            self.reduction_partials_[reduction.symbol] = self._partials(reduction.symbols, [reduction.argument], by)
        self.operand_partials_: dict[int, PartialsSpec] = {}

    def _partials(
        self,
        arguments: list[Expression],
        expressions: list[Expression],
        by: list[Expression],
    ) -> PartialsSpec:
        """Return the nonzero partial derivatives of expressions by symbols, compiled into one NumPy function."""
        pairs = tuple(
            (index, item)
            for index, expression in enumerate(expressions)
            for item in by
            if item in SR(expression).variables()
        )
        derivatives = [SR(expressions[index]).diff(item) for index, item in pairs]
        if not derivatives:
            return PartialsSpec(pairs, None)
        printer = ExpressionPrinter(time=self.time, compiled=True).prepare(arguments)
        return PartialsSpec(pairs, ExpressionPrinter._function_of(printer.run(derivatives).code_))

    @staticmethod
    def _derivatives(
        partials: PartialsSpec,
        t: float,
        values: Sequence[object],
    ) -> list[tuple[int, Expression, npt.ArrayLike]]:
        """Return every derivative with the index of its expression and the symbol it is taken by."""
        if partials.function is None:
            return []
        results = ExpressionPrinter._values_of(partials.function(t, *values))
        return [(index, item, value) for (index, item), value in zip(partials.pairs, results, strict=True)]

    @property
    def _by_symbol(self) -> dict[Expression, ReductionKernel]:
        """Reductions by placeholder symbol."""
        return {item.symbol: item for item in self.reductions.reductions}

    def _member_gradients(
        self,
        reduction: ReductionKernel,
        t: float,
        x: Array,
        q: Array,
        gradients: dict[Expression, Sparse],
    ) -> Sparse:
        """Return the gradient of every reduced member's argument by the state, one row per member."""
        total = scipy.sparse.csr_array((reduction.size, self.size))
        members = np.arange(reduction.size)
        for _, item, value in self._derivatives(self.reduction_partials_[reduction.symbol], t, reduction.gather(x, q)):
            derivative = np.broadcast_to(np.asarray(value, dtype=float), (reduction.size,))
            read = reduction.reads[item]
            if item in gradients:
                nested = gradients[item]
                block = nested if nested.shape[0] == reduction.size else scipy.sparse.vstack([nested] * reduction.size)
                total = total + scipy.sparse.diags_array(derivative) @ block
                continue
            columns = np.broadcast_to(np.asarray(read.index, dtype=np.intp), (reduction.size,))
            total = total + self._selection(members, columns, derivative, self.size)
        return scipy.sparse.csr_array(total)

    def _reduction_gradient(
        self,
        reduction: ReductionKernel,
        t: float,
        x: Array,
        q: Array,
        gradients: dict[Expression, Sparse],
    ) -> Sparse:
        """Return the gradient of a reduction by the state, one row per reading member for weights."""
        argument, condition = reduction.values(t, x, q)
        mask = condition.astype(float)
        selected = float(mask.sum())
        members = self._member_gradients(reduction, t, x, q, gradients)
        if reduction.kind == 'count' or not selected:
            return scipy.sparse.csr_array((reduction.readers, self.size))
        if reduction.weights is not None:
            weighted = reduction.weight_matrix(x, q) @ scipy.sparse.diags_array(mask) @ members
            return scipy.sparse.csr_array(weighted / selected if reduction.kind == 'mean' else weighted)
        if reduction.kind in {'max', 'min'}:
            values = np.where(condition, argument, -np.inf if reduction.kind == 'max' else np.inf)
            first = int(np.argmax(values) if reduction.kind == 'max' else np.argmin(values))
            return scipy.sparse.csr_array(members[[first]])
        if reduction.kind == 'var':
            mean = float(argument[condition].mean())
            factors = 2 * mask * (argument - mean) / selected
        elif reduction.kind == 'logsumexp':
            factors = mask * np.exp(argument - logsumexp(argument[condition]))
        else:
            factors = mask / selected if reduction.kind == 'mean' else mask
        return scipy.sparse.csr_array(factors[np.newaxis, :] @ members)

    def _reduced_rows(self, kernel: LawKernel, reference: ReferenceSpec, rows: Sparse) -> Sparse:
        """Return the rows of a reduction's gradient or support read by every evaluation index of a kernel."""
        if reference.entry.members is None:
            rows = rows[[min(reference.column, rows.shape[0] - 1)]]
        return rows if rows.shape[0] == kernel.rows else scipy.sparse.vstack([rows] * kernel.rows, format='csr')

    def _symbol_gradient(
        self,
        kernel: LawKernel,
        item: Expression,
        *,
        t: float,
        x: Array,
        q: Array,
        gradients: dict[Expression, Sparse],
    ) -> Sparse | None:
        """Return the gradient of a symbol by the state, one row per evaluation index, or `None` for parameters."""
        reference = kernel.references[item]
        rows = np.arange(kernel.rows)
        if reference.source is ReadKind.REDUCED:
            return self._reduced_rows(kernel, reference, gradients[item])
        if reference.source is ReadKind.DERIVED:
            read = reference.derived
            key = id(read)
            if key not in self.operand_partials_:
                by = [symbol for symbol in read.symbols if self._is_variable(kernel.references[symbol])]
                self.operand_partials_[key] = self._partials(read.symbols, [read.operand], by)
            values = [kernel.read(symbol, t, x, q) for symbol in read.symbols]
            total = scipy.sparse.csr_array((kernel.rows, self.size))
            for _, symbol, value in self._derivatives(self.operand_partials_[key], t, values):
                inner = self._symbol_gradient(kernel, symbol, t=t, x=x, q=q, gradients=gradients)
                if inner is not None:
                    derivative = np.broadcast_to(np.asarray(value, dtype=float), (kernel.rows,))
                    total = total + scipy.sparse.diags_array(derivative) @ inner
            return scipy.sparse.csr_array(read.matrix @ total)
        if not reference.is_state:
            return None
        if reference.source is ReadKind.NEIGHBOR:
            members, columns = self._neighbor_columns(reference)
            return self._selection(members, columns, 1.0, self.size, height=kernel.rows)
        return self._selection(
            rows,
            LawKernel._member_indices(reference, kernel.rows),
            1.0,
            self.size,
            height=kernel.rows,
        )

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the Jacobian of the drift at the state as the sparse matrix of `dF_i/dx_j`."""
        state = self._state_of(inputs)
        t, x, p = state.t, state.x, state.p
        partials = self.kernel_partials_
        q = self.reductions.extend(t, x, p)
        gradients: dict[Expression, Sparse] = {}
        for reduction in self.reductions.reductions:
            gradients[reduction.symbol] = self._reduction_gradient(reduction, t, x, q, gradients)
        jacobian = scipy.sparse.csr_array((self.size, self.size))
        for kernel, kernel_partials in zip(self.kernels, partials, strict=True):
            for index, item, value in self._derivatives(kernel_partials, t, kernel.gather(t, x, q)):
                gradient = self._symbol_gradient(kernel, item, t=t, x=x, q=q, gradients=gradients)
                if gradient is None:
                    continue
                target = LawKernel._member_indices(self._target_of(kernel.outputs[index]), kernel.rows)
                scatter = self._selection(np.arange(kernel.rows), target, 1.0, self.size).T
                derivative = np.broadcast_to(np.asarray(value, dtype=float), (kernel.rows,))
                jacobian = jacobian + scatter @ scipy.sparse.diags_array(derivative) @ gradient
        for projection in self._mirror_projections(self.symmetries, self.size):
            jacobian = projection @ jacobian
        self.values_ = scipy.sparse.csr_array(self._free_rows() @ jacobian)

    def _free_rows(self) -> Sparse:
        """Return the diagonal matrix that empties the fixed rows."""
        diagonal = np.ones(self.size)
        if self.constrained is not None:
            diagonal[self.constrained] = 0.0
        return scipy.sparse.diags_array(diagonal, format='csr')

    def pattern(self) -> Sparse | None:
        """Return the boolean structure of the Jacobian, or `None` when its dense blocks are too large to state.

        Returns:
            pattern:
                A boolean sparse matrix that holds every entry the Jacobian can make nonzero and the diagonal of free
                rows, with fixed rows empty, or `None` when the dense blocks exceed `_PATTERN_LIMIT` entries or the
                laws reduce over neighbourhoods, whose neighbours can change.
        """
        return self._pattern

    @functools.cached_property
    def _pattern(self) -> Sparse | None:
        """Boolean structure of the Jacobian, built once, or `None` when it is too dense to state."""
        try:
            return self._build_pattern()
        except _TooDenseError:
            return None

    def _build_pattern(self) -> Sparse:
        """Return the boolean structure of the Jacobian, raising `_TooDenseError` above the size limit."""
        self._dense = 0
        if self.reductions.neighbourhoods:
            raise _TooDenseError
        supports: dict[Expression, Sparse] = {}
        for reduction in self.reductions.reductions:
            supports[reduction.symbol] = self._reduction_support(reduction, supports)
        blocks = [scipy.sparse.eye_array(self.size, format='csr')]
        for kernel in self.kernels:
            blocks.extend(self._kernel_pattern(kernel, supports))
        blocks.extend(volume.pattern(self.size) for volume in self.volumes)
        total = scipy.sparse.csr_array((self.size, self.size))
        for block in blocks:
            total = total + scipy.sparse.csr_array(block, dtype=float)
        pattern = self._boolean(total)
        for projection in self._mirror_projections(self.symmetries, self.size):
            pattern = self._product(self._boolean(projection), pattern)
        return self._product(self._free_rows(), pattern)

    def _note_dense(self, entries: int) -> None:
        """Count the entries of a dense block and stop above the size limit."""
        self._dense += entries
        if self._dense > _PATTERN_LIMIT:
            raise _TooDenseError

    def _dense_rows(self, rows: int, columns: Indices) -> Sparse:
        """Return the block where every evaluation index reads every one of the columns."""
        columns = np.unique(columns)
        self._note_dense(rows * columns.size)
        return self._selection(np.repeat(np.arange(rows), columns.size), np.tile(columns, rows), 1.0, self.size, rows)

    def _all_columns(self, kernel: LawKernel, item: Expression, supports: dict[Expression, Sparse]) -> Indices:
        """Return every state index a symbol can read at any evaluation index."""
        reference = kernel.references[item]
        if reference.source is ReadKind.REDUCED:
            return np.unique(supports[item].indices).astype(np.intp)
        if reference.source is ReadKind.DERIVED:
            parts = [self._all_columns(kernel, symbol, supports) for symbol in reference.derived.symbols]
            return np.unique(np.concatenate([np.zeros(0, dtype=np.intp), *parts]))
        if not reference.is_state:
            return np.zeros(0, dtype=np.intp)
        if reference.source is ReadKind.SINGLE:
            return np.array([reference.index], dtype=np.intp)
        return reference.flat_indices.astype(np.intp)

    def _symbol_support(self, kernel: LawKernel, item: Expression, supports: dict[Expression, Sparse]) -> Sparse | None:
        """Return the states a symbol reads at every evaluation index of a member kernel, or `None` for parameters."""
        reference = kernel.references.get(item)
        if reference is None:
            return None
        if reference.source is ReadKind.REDUCED:
            block = self._reduced_rows(kernel, reference, supports[item])
            self._note_dense(block.nnz)
            return block
        if reference.source is ReadKind.DERIVED:
            return self._derived_support(kernel, item, supports)
        if not reference.is_state:
            return None
        return self._state_support(kernel, reference)

    def _derived_support(self, kernel: LawKernel, item: Expression, supports: dict[Expression, Sparse]) -> Sparse:
        """Return the states a derived read reads, through its sparse operator or as a dense block."""
        read = kernel.references[item].derived
        if read.matrix is None:
            return self._dense_rows(kernel.rows, self._all_columns(kernel, item, supports))
        operand = scipy.sparse.csr_array((kernel.rows, self.size))
        for symbol in read.symbols:
            block = self._symbol_support(kernel, symbol, supports)
            if block is not None:
                operand = operand + scipy.sparse.csr_array(block, dtype=float)
        return self._product(self._boolean(read.matrix), operand)

    def _state_support(self, kernel: LawKernel, reference: ReferenceSpec) -> Sparse:
        """Return the states a state symbol reads at every evaluation index."""
        rows = kernel.rows
        if reference.source is ReadKind.NEIGHBOR:
            members, columns = self._neighbor_columns(reference)
            return self._selection(members, columns, 1.0, self.size, height=rows)
        if reference.source is ReadKind.SINGLE:
            return self._selection(np.arange(rows), np.full(rows, reference.index, dtype=np.intp), 1.0, self.size, rows)
        if reference.source is ReadKind.MEMBER and reference.entry.rows == rows:
            return self._selection(np.arange(rows), reference.flat_indices, 1.0, self.size, height=rows)
        return self._dense_rows(rows, reference.flat_indices)

    def _kernel_pattern(self, kernel: LawKernel, supports: dict[Expression, Sparse]) -> list[Sparse]:
        """Return the blocks of the pattern that the outputs of one kernel add."""
        blocks = []
        everything = np.arange(self.size, dtype=np.intp)
        for output in kernel.outputs:
            target = self._target_of(output)
            used = self._output_symbols(output)
            if kernel.pairs is not None or target.source in {ReadKind.FIRST, ReadKind.SECOND}:
                targets = np.unique(target.flat_indices if target.source is not ReadKind.SINGLE else [target.index])
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
                blocks.append(self._selection(targets, np.arange(targets.size), 1.0, targets.size, self.size) @ block)
                continue
            scatter = self._selection(
                np.arange(kernel.rows),
                LawKernel._member_indices(target, kernel.rows),
                1.0,
                self.size,
            ).T
            if used is None:
                blocks.append(scatter @ self._dense_rows(kernel.rows, everything))
                continue
            for item in used:
                support = self._symbol_support(kernel, item, supports)
                if support is not None:
                    blocks.append(scatter @ support)
        return blocks

    def _reduction_support(self, reduction: ReductionKernel, supports: dict[Expression, Sparse]) -> Sparse:
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
                support = support + (self._selection(members, columns, 1.0, self.size) != 0)
        if reduction.kind == 'count':
            return scipy.sparse.csr_array((1, self.size), dtype=bool)
        if reduction.weights is not None:
            weights = reduction.weights
            pattern = np.ones(weights.entry.shape) if isinstance(weights, ReadSpec) and weights.entry else weights
            return scipy.sparse.csr_array((abs(pattern) != 0).astype(float) @ support.astype(float) != 0)
        return scipy.sparse.csr_array(np.asarray(support.astype(float).sum(axis=0)).reshape(1, -1) != 0)

    @staticmethod
    def _selection(
        rows: npt.NDArray[np.intp],
        columns: npt.NDArray[np.intp],
        values: npt.ArrayLike,
        size: int,
        height: int | None = None,
    ) -> Sparse:
        """Return the sparse matrix with the given values at the given rows and columns, summing duplicates."""
        shape = (int(rows.max(initial=-1)) + 1 if height is None else height, size)
        data = np.broadcast_to(np.asarray(values, dtype=float), rows.shape)
        return scipy.sparse.csr_array((data, (rows, columns)), shape=shape)

    @staticmethod
    def _product(first: Sparse, second: Sparse) -> Sparse:
        """Return the structure of the product of two boolean structures."""
        return JacobianKernel._boolean(
            scipy.sparse.csr_array(first, dtype=float) @ scipy.sparse.csr_array(second, dtype=float),
        )

    @staticmethod
    def _target_of(output: OutputSpec) -> ReferenceSpec:
        """Return the target component of a rate output, raising an error for an output without one.

        Args:
            output:
                The output.

        Returns:
            target:
                The target component.

        Raises:
            ValidationError:
                The output has no target, as the propensity of a jump.
        """
        if output.target is None:
            error_msg = 'A rate output of a Jacobian has no target component.'
            raise ValidationError(error_msg)
        return output.target

    @staticmethod
    def _output_symbols(output: OutputSpec) -> set[Expression] | None:
        """Return the symbols an output reads, or `None` for a callable law, which reads every state."""
        if output.expression is not None:
            return set(SR(output.expression).variables())
        if output.arguments is not None:
            return symbols_of(list(output.arguments))
        return None

    @staticmethod
    def _neighbor_columns(reference: ReferenceSpec) -> tuple[Indices, Indices]:
        """Return the members that read a neighbour and the flat index of the neighbour each reads."""
        table = np.asarray(reference.shifted.table, dtype=np.intp)
        entry = reference.entry
        members = np.arange(table.size, dtype=np.intp)
        inside = table < entry.rows
        columns = entry.start + table[inside] * entry.columns + reference.column
        return members[inside], columns.astype(np.intp)

    @staticmethod
    def _mirror_projections(symmetries: Sequence[SymmetrySpec], size: int) -> list[Sparse]:
        """Return the matrices that project the rates of symmetric states onto their odd or even parts, axis by axis."""
        projections = []
        for symmetry in symmetries:
            indices = symmetry.indices
            grid = np.arange(int(np.prod(symmetry.shape))).reshape(symmetry.shape)
            free = np.setdiff1d(np.arange(size), indices)
            for axis, sign in symmetry.signs:
                mirror = np.roll(np.flip(grid, axis=axis), 1, axis=axis).ravel()
                rows = np.concatenate([free, indices, indices])
                columns = np.concatenate([free, indices, indices[mirror]])
                values = np.concatenate(
                    [np.ones(free.size), np.full(indices.size, 0.5), np.full(indices.size, sign / 2)],
                )
                projections.append(scipy.sparse.csr_array((values, (rows, columns)), shape=(size, size)))
        return projections

    @staticmethod
    def _is_variable(reference: ReferenceSpec) -> bool:
        """Check whether a reference is a state, a reduction, or a derived read, so derivatives reach the state."""
        return reference.source in {ReadKind.REDUCED, ReadKind.DERIVED} or reference.is_state

    @staticmethod
    def _boolean(matrix: scipy.sparse.sparray) -> Sparse:
        """Return the nonzero structure of a sparse matrix as a boolean CSR matrix."""
        matrix = scipy.sparse.csr_array(matrix)
        matrix.eliminate_zeros()
        return scipy.sparse.csr_array(
            (np.ones(matrix.nnz, dtype=bool), matrix.indices, matrix.indptr),
            shape=matrix.shape,
        )
