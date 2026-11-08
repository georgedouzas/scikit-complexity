"""Reduction that laws read, compiled into a function of the values of its symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression
from scipy.special import logsumexp

from ...errors import ValidationError
from ...specs import EntrySpec, ReadSpec, RunStateSpec
from ...types import AnyArray, Array
from ._base import BaseKernel
from ._expression_printer import ExpressionPrinter

REDUCTIONS: dict[str, Callable[[AnyArray], npt.ArrayLike]] = {
    'count': lambda array: np.full(array.shape[1:], float(array.shape[0])),
    'sum': lambda array: array.sum(axis=0),
    'mean': lambda array: array.mean(axis=0),
    'min': lambda array: array.min(axis=0),
    'max': lambda array: array.max(axis=0),
    'var': lambda array: array.var(axis=0),
    'logsumexp': lambda array: logsumexp(array, axis=0),
}


class ReductionKernel(BaseKernel):
    """Reduction that laws read, compiled into a function of the values of its symbols.

    Args:
        symbol:
            The placeholder symbol that kernels read instead of the reduction.

        kind:
            `'sum'`, `'mean'`, `'max'`, `'min'`, `'var'`, `'logsumexp'`, or `'count'`.

        argument:
            The expression of one reduced member, with nested reductions replaced by their placeholders.

        condition:
            The condition that selects the members, or `None` for all members.

        weights:
            The constant weight matrix as a CSR array of shape `(readers, size)`, the read of a shaped parameter of
            that shape, or `None`.

        entry:
            The hidden entry that holds the value, after the parameters in the extended parameter vector.

        size:
            The number of reduced members.

        group:
            The path of the reduced group, the labels of an element set, or `None` for no group.

        reads:
            A mapping from every symbol of the argument and the condition to its values.

        time:
            The time symbol.

        members:
            The entry of the reduced group that gives its size to compiled loops, or `None` for an element set.

        function:
            The function of the array of member values of a reduction of the kind `'function'`, or `None`.
    """

    values_: None

    def __init__(
        self,
        symbol: Expression,
        kind: str,
        argument: Expression,
        condition: Expression | None,
        weights: scipy.sparse.csr_array | ReadSpec | None,
        *,
        entry: EntrySpec,
        size: int,
        group: str | None,
        reads: dict[Expression, ReadSpec],
        time: Expression,
        members: EntrySpec | None = None,
        function: Callable[[AnyArray], object] | None = None,
    ) -> None:
        self.symbol = symbol
        self.kind = kind
        self.argument = argument
        self.condition = condition
        self.weights = weights
        self.entry = entry
        self.size = size
        self.group = group
        self.reads = reads
        self.time = time
        self.members = members
        self.function = function

    @functools.cached_property
    def _symbols(self) -> list[Expression]:
        """Symbols the argument and the condition read, sorted by name."""
        return sorted(self.reads, key=str)

    @functools.cached_property
    def _function(self) -> Callable[..., object]:
        """NumPy function of the argument and the condition, compiled once."""
        expressions = [self.argument, self.condition if self.condition is not None else 1]
        printer = ExpressionPrinter(time=self.time, compiled=True).prepare(self._symbols)
        return ExpressionPrinter._function_of(printer.run(expressions).code_)

    @property
    def symbols(self) -> list[Expression]:
        """Symbols the argument and the condition read.

        They come in the order the compiled function takes them after time.
        """
        return list(self._symbols)

    @property
    def readers(self) -> int:
        """Number of reading members of a weighted reduction, 1 without weights."""
        return 1 if self.weights is None else int(self.weights.shape[0])

    def weight_matrix(self, x: Array, q: Array) -> scipy.sparse.csr_array | Array:
        """Return the weight matrix.

        It is read from the vectors for the weights of a shaped parameter.

        Args:
            x:
                The state vector.

            q:
                The extended parameter vector.

        Returns:
            weights:
                The weight matrix, one row per reading member and one column per reduced member.

        Raises:
            ValidationError:
                The reduction has no weights.
        """
        weights = self.weights
        if weights is None:
            error_msg = f'The reduction {self.symbol} has no weights.'
            raise ValidationError(error_msg)
        if isinstance(weights, ReadSpec) and weights.entry is not None:
            return (x if weights.is_state else q)[weights.index].reshape(weights.entry.shape)
        return weights

    def gather(self, x: Array, q: Array) -> list[npt.ArrayLike]:
        """Return the values of the symbols the reduction reads.

        Args:
            x:
                The state vector.

            q:
                The extended parameter vector.

        Returns:
            values:
                One value or one array of member values per symbol of `symbols`.
        """
        return [(x if read.is_state else q)[read.index] for read in (self.reads[item] for item in self._symbols)]

    def values(self, t: float, x: Array, q: Array) -> tuple[Array, npt.NDArray[np.bool_]]:
        """Return the argument and the condition of every reduced member.

        Args:
            t:
                The time.

            x:
                The state vector.

            q:
                The extended parameter vector.

        Returns:
            values:
                The argument and the condition, one value per reduced member.
        """
        argument, condition = ExpressionPrinter._values_of(self._function(t, *self.gather(x, q)))
        return (
            np.broadcast_to(np.asarray(argument, dtype=float), (self.size,)),
            np.broadcast_to(np.asarray(condition, dtype=bool), (self.size,)),
        )

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Write the value of the reduction into the extended parameter vector.

        The vector is the `p` of `inputs`.
        """
        self.values_ = None
        state = self._state_of(inputs)
        q = state.p
        argument, condition = self.values(state.t, state.x, q)
        if self.weights is None:
            q[self.entry.start] = self._reduce_values(argument[condition], self.function or self.kind)
            return
        weighted = self.weight_matrix(state.x, q) @ np.where(condition, argument, 0.0)
        if self.kind == 'mean':
            selected = int(np.count_nonzero(condition))
            weighted = weighted / selected if selected else np.zeros_like(weighted)
        q[self.entry.indices] = weighted

    @staticmethod
    def _reduce_values(
        values: npt.ArrayLike | Sequence[object],
        reduce: str | Callable[[AnyArray], object],
    ) -> npt.ArrayLike:
        """Return the reduction of the values of a group's members.

        Args:
            values:
                One value per member, or one row per member for vector values.

            reduce:
                `'count'`, `'sum'`, `'mean'`, `'min'`, `'max'`, `'var'`, `'logsumexp'`, or a function of the array of
                values.

        Returns:
            value:
                The reduction, a number for scalar values and an array for vector values. The reductions of no members
                are 0, except the count.

        Raises:
            ValidationError:
                The reduction is unknown, or the reduction function returns neither a number nor an array.

        Examples:
            >>> from skcomplex.core.engine.problems._reduction_kernel import ReductionKernel
            >>> ReductionKernel._reduce_values([1.0, 2.0, 6.0], 'mean')
            3.0
        """
        array = np.asarray(values)
        array = np.zeros(len(array)) if reduce == 'count' else array
        if callable(reduce):
            reduced = reduce(array)
            if not isinstance(reduced, int | float | complex | np.number | np.ndarray):
                error_msg = f'The reduction function returned a {type(reduced).__name__}, not a number or an array.'
                raise ValidationError(error_msg)
            return reduced
        if reduce not in REDUCTIONS:
            error_msg = f'Unknown reduction {reduce!r}. Use {", ".join(REDUCTIONS)}, or a function of the values.'
            raise ValidationError(error_msg)
        if not array.size and reduce != 'count':
            return 0.0
        result = np.real_if_close(REDUCTIONS[reduce](array))
        if np.ndim(result):
            return result
        return float(result) if np.isrealobj(result) else complex(result)
