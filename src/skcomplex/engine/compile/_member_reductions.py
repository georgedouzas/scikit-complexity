"""Reductions in the member expression of an observable, compiled with the reduction lowering of laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Self

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import RR, SR, Expression, I

from ...base import BaseConfig
from ...core.laws.utils import reductions_in
from ...errors import RoleError, ValidationError
from ...specs import EntrySpec, ReadSpec, ReductionSpec
from .._utils import symbol, symbols_of
from ..problems._reduction_kernel import ReductionKernel


class _MemberReductions(BaseConfig[None, object]):
    """Reductions in the member expression of an observable, compiled with the reduction lowering of laws.

    `prepare()` starts with the reads given and no reduction found, and `run(expression)` returns an expression with
    every reduction replaced by the placeholder of its value, compiling every reduction once.

    Args:
        path:
            The path of the observable.

        group:
            The path of the group the observable reduces.

        size:
            The number of members of the group.

        reads:
            How the reduction pass reads every symbol of the expression.

        time:
            The time symbol.

        parameters:
            The size of the parameter vector.

    Attributes:
        reads_:
            The reads, with the placeholder of every reduction found.

        found_:
            The kernel of every reduction found, by its call.
    """

    reads_: dict[Expression, ReadSpec]
    found_: dict[str, ReductionKernel]

    def __init__(
        self,
        *,
        path: str,
        group: str,
        size: int,
        reads: Mapping[Expression, ReadSpec],
        time: Expression,
        parameters: int,
    ) -> None:
        self.path = path
        self.group = group
        self.size = size
        self.reads = reads
        self.time = time
        self.parameters = parameters

    def _prepare(self, context: None) -> Self:
        """Start with the reads given and no reduction found.

        Args:
            context:
                Nothing: the lowering serves its observable.

        Returns:
            self:
                The prepared lowering.
        """
        self.reads_, self.found_ = dict(self.reads), {}
        return self

    def _run(self, inputs: object | None) -> object:
        """Return an expression with every reduction replaced by the placeholder of its value.

        Args:
            inputs:
                The expression, or any other value, which is returned unchanged.

        Returns:
            lowered:
                The expression with placeholders.
        """
        if not isinstance(inputs, Expression):
            return inputs
        placeholders = {item.call: self._placeholder(item) for item in reductions_in(inputs)}
        lowered = inputs
        for call in sorted(placeholders, key=lambda item: -len(str(item))):
            lowered = lowered.subs({call: placeholders[call]})
        return lowered

    @property
    def extent(self) -> int:
        """Size of the parameter vector extended by the values of the reductions."""
        return max((item.entry.start + item.entry.size for item in self.found_.values()), default=self.parameters)

    def _check_over(self, reduction: ReductionSpec) -> None:
        """Raise an error when a reduction reduces over a group other than the observable's group."""
        read = self.reads_.get(reduction.over) if reduction.over is not None else None
        if reduction.over is None or (read is not None and read.per_member):
            return
        error_msg = (
            f'The reduction {reduction.call} of observable {self.path} reads a group other than {self.group}. A '
            'reduction in the member expression of an observable reduces over the observable\'s group, so write one '
            'observable per group.'
        )
        raise RoleError(error_msg)

    def _read(self, item: Expression) -> ReadSpec:
        """Return how the reduction pass reads one symbol of a reduction."""
        if item not in self.reads_:
            error_msg = (
                f'A reduction of observable {self.path} reads {item}, which is not a variable of a member of '
                f'{self.group}, of a single element, or another reduction. Reductions read the namespace the '
                'expression receives.'
            )
            raise ValidationError(error_msg)
        return self.reads_[item]

    def _weights(self, reduction: ReductionSpec) -> scipy.sparse.csr_array | None:
        """Return the weights of a reduction as a CSR array after checking their shape."""
        if reduction.weights is None:
            return None
        weights = reduction.weights
        if isinstance(weights, scipy.sparse.sparray | scipy.sparse.spmatrix):
            source: scipy.sparse.sparray | scipy.sparse.spmatrix | np.ndarray = weights
        else:
            source = np.asarray(weights, dtype=float)
            if source.ndim == 1 and source.size == self.size:
                source = np.broadcast_to(source, (self.size, self.size))
        shape = tuple(source.shape)
        if shape != (self.size, self.size):
            error_msg = (
                f'The weights of the reduction {reduction.call} of observable {self.path} have the shape {shape}, but '
                f'the reduction needs the shape {(self.size, self.size)}, one row and one column per member of '
                f'{self.group}.'
            )
            raise ValidationError(error_msg)
        return scipy.sparse.csr_array(source, dtype=float)

    def _placeholder(self, reduction: ReductionSpec) -> Expression:
        """Return the placeholder of a reduction, compiling it the first time it is found."""
        key = str(reduction.call)
        if key in self.found_:
            return self.found_[key].symbol
        if SR(reduction.argument).has(I):
            error_msg = (
                f'The reduction {reduction.call} of observable {self.path} reduces a complex expression. Reduce its '
                'real or imaginary part.'
            )
            raise ValidationError(error_msg)
        self._check_over(reduction)
        argument = SR(self._run(reduction.argument))
        condition = None if reduction.condition is None else SR(self._run(reduction.condition))
        symbols = symbols_of([argument, *([condition] if condition is not None else [])]) - {self.time}
        weights = self._weights(reduction)
        index = len(self.found_)
        shape = (self.size,) if weights is not None else ()
        members = self.size if weights is not None else None
        entry = EntrySpec(f'__observed_{index}', self.extent, shape, RR, members)
        placeholder = symbol(f'_observed_{index}')
        reads = {item: self._read(item) for item in sorted(symbols, key=str)}
        self.found_[key] = ReductionKernel(
            symbol=placeholder,
            kind=reduction.kind,
            argument=argument,
            condition=condition,
            weights=weights,
            entry=entry,
            size=self.size,
            group=self.group,
            reads=reads,
            time=self.time,
        ).check()
        indices = entry.start + np.arange(self.size) if weights is not None else entry.start
        self.reads_[placeholder] = ReadSpec(False, indices, entry)
        return placeholder
