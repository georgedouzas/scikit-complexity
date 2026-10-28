"""Reduction that laws read, compiled into a function of the values of its symbols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression

from ...core.errors import ValidationError
from ...core.types import Indices, Vector
from ._entry import Entry
from ._printing import numpy_function
from ._reads import Read

REDUCTIONS: dict[str, Callable[[Any], Any]] = {
    'count': lambda array: np.full(array.shape[1:], float(array.shape[0])),
    'sum': lambda array: array.sum(axis=0),
    'mean': lambda array: array.mean(axis=0),
    'min': lambda array: array.min(axis=0),
    'max': lambda array: array.max(axis=0),
    'var': lambda array: array.var(axis=0),
}


@dataclass
class CompiledReduction:
    """Reduction that laws read, compiled into a function of the values of its symbols.

    Args:
        symbol:
            The placeholder symbol that kernels read instead of the reduction.

        kind:
            `'sum'`, `'mean'`, `'max'`, `'min'`, `'var'`, or `'count'`.

        argument:
            The expression of one reduced member, with nested reductions replaced by their placeholders.

        condition:
            The condition that selects the members, or `None` for all members.

        weights:
            The constant weight matrix as a CSR array of shape `(readers, size)`, or `None`.

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

        set_indices:
            The flat state indices of the elements of an element set, one per element, or `None`.
    """

    symbol: Expression
    kind: str
    argument: Expression
    condition: Expression | None
    weights: scipy.sparse.csr_array | None
    entry: Entry
    size: int
    group: str | None
    reads: dict[Expression, Read]
    time: Expression
    members: Entry | None = None
    set_indices: Indices | None = None

    @functools.cached_property
    def _symbols(self) -> list[Expression]:
        """Symbols the argument and the condition read, sorted by name."""
        return sorted(self.reads, key=str)

    @functools.cached_property
    def _function(self) -> Callable[..., list[Any]]:
        """NumPy function of the argument and the condition, compiled once."""
        expressions = [self.argument, self.condition if self.condition is not None else 1]
        return numpy_function(self.time, self._symbols, expressions)

    @property
    def symbols(self) -> list[Expression]:
        """Symbols the argument and the condition read, in the order the compiled function takes them after time."""
        return list(self._symbols)

    def gather(self, x: Vector, q: Vector) -> list[Any]:
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

    def values(self, t: float, x: Vector, q: Vector) -> tuple[Vector, npt.NDArray[np.bool_]]:
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
        argument, condition = self._function(t, *self.gather(x, q))
        return (
            np.broadcast_to(np.asarray(argument, dtype=float), (self.size,)),
            np.broadcast_to(np.asarray(condition, dtype=bool), (self.size,)),
        )

    def evaluate(self, t: float, x: Vector, q: Vector) -> None:
        """Write the value of the reduction into the extended parameter vector.

        Args:
            t:
                The time.

            x:
                The state vector.

            q:
                The extended parameter vector, which holds the parameters and the reductions before this one.
        """
        argument, condition = self.values(t, x, q)
        if self.weights is None:
            q[self.entry.start] = self._reduce_values(argument[condition], self.kind)
            return
        weighted = self.weights @ np.where(condition, argument, 0.0)
        if self.kind == 'mean':
            selected = int(np.count_nonzero(condition))
            weighted = weighted / selected if selected else np.zeros_like(weighted)
        q[self.entry.indices] = weighted

    @staticmethod
    def _reduce_values(values: Any, reduce: str | Callable[[Any], Any]) -> Any:  # noqa: ANN401
        """Return the reduction of the values of a group's members.

        Args:
            values:
                One value per member, or one row per member for vector values.

            reduce:
                `'count'`, `'sum'`, `'mean'`, `'min'`, `'max'`, `'var'`, or a function of the array of values.

        Returns:
            value:
                The reduction, a number for scalar values and an array for vector values. The reductions of no members
                are 0, except the count.

        Raises:
            ValidationError:
                The reduction is unknown.

        Examples:
            >>> from skcomplex.engine.problems._compiled_reduction import CompiledReduction
            >>> CompiledReduction._reduce_values([1.0, 2.0, 6.0], 'mean')
            3.0
        """
        array = np.asarray(values) if reduce != 'count' else np.zeros(len(values))
        if callable(reduce):
            return reduce(array)
        if reduce not in REDUCTIONS:
            error_msg = f'Unknown reduction {reduce!r}. Use {", ".join(REDUCTIONS)}, or a function of the values.'
            raise ValidationError(error_msg)
        if not array.size and reduce != 'count':
            return 0.0
        result = np.real_if_close(REDUCTIONS[reduce](array))
        if np.ndim(result):
            return result
        return float(result) if np.isrealobj(result) else complex(result)
