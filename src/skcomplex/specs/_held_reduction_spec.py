"""Define the held reduction spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ..errors import RoleError, ValidationError
from ..types import Array
from ._base_spec import BaseSpec
from ._reduction_spec import ReductionSpec


@dataclass(frozen=True)
class _HeldReductionSpec(BaseSpec):
    """Reduction of a group that a co-simulation computes at every synchronisation point and holds until the next."""

    group: str
    reduce: str | Callable[[Any], Any]
    variables: tuple[str, ...] = ()
    symbols: tuple[Expression, ...] = ()
    argument: Expression | None = None
    condition: Expression | None = None

    @property
    def key(self) -> tuple[str, ...]:
        """Group, reduction, argument, and condition, which identify the reduction."""
        return (self.group, str(self.reduce), str(self.argument), str(self.condition))

    def reducer(self, time: Expression) -> Callable[[float, Sequence[Array], Sequence[Any]], float]:
        """Return the function of time, the values of the variables, and the members that computes the reduction.

        Args:
            time:
                The time symbol.

        Returns:
            reducer:
                The function.
        """
        from ..engine.problems import ExpressionPrinter
        from ..engine.problems._reduction_kernel import ReductionKernel
        reduce = self.reduce
        if self.argument is None:

            def reduce_values_of(t: float, columns: Sequence[Array], members: Sequence[Any]) -> float:
                from ..engine.problems._reduction_kernel import ReductionKernel
                del t
                return float(np.real(ReductionKernel._reduce_values(columns[0] if columns else members, reduce)))

            return reduce_values_of
        condition = SR(1) if self.condition is None else self.condition
        printer = ExpressionPrinter(time=time, compiled=True).check().prepare(list(self.symbols))
        function = printer.run([self.argument, condition])

        def reduce_expression(t: float, columns: Sequence[Array], members: Sequence[Any]) -> float:
            size = len(members)
            argument, selected = (
                np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in function(t, *columns)
            )
            return float(np.real(ReductionKernel._reduce_values(argument[selected.astype(bool)], reduce)))

        return reduce_expression

    @staticmethod
    def _held_reduction(
        reduction: ReductionSpec,
        agents: Mapping[Expression, tuple[str, str]],
        time: Expression,
        owner: str,
    ) -> '_HeldReductionSpec | None':
        """Return the held reduction of a reduction of agents, or `None` for a reduction of elements.

        Args:
            reduction:
                The reduction a law reads.

            agents:
                The path of the agent group and the variable of every symbol of agents.

            time:
                The time symbol.

            owner:
                The path of the component whose law reads the reduction.

        Returns:
            held:
                The held reduction, or `None`.

        Raises:
            RoleError:
                The reduction reads the agents of more than one group.

            ValidationError:
                The reduction has weights, or reads a symbol that is not a variable of the agents.
        """
        from ..engine._utils import symbols_of
        expressions = [SR(reduction.argument), *([SR(reduction.condition)] if reduction.condition is not None else [])]
        symbols = symbols_of(expressions) - {time}
        read = symbols | ({reduction.over} if reduction.over is not None else set())
        groups = {agents[item][0] for item in read if item in agents}
        if not groups:
            return None
        if len(groups) > 1:
            error_msg = (
                f'The reduction {reduction.call} of {owner} reads the agents of {" and ".join(sorted(groups))}. A '
                'reduction reduces over one group, so write one reduction per group.'
            )
            raise RoleError(error_msg)
        group = next(iter(groups))
        others = sorted(str(item) for item in symbols if item not in agents or agents[item][0] != group)
        if reduction.weights is not None or others:
            error_msg = (
                f'The reduction {reduction.call} of {owner} reduces the agents of {group}, which a co-simulation '
                'computes at its synchronisation points from the scalar states and parameters of the agents alone, '
                f'without weights{", but it reads " + ", ".join(others) if others else ""}. Reduce an expression of '
                'the agents.'
            )
            raise ValidationError(error_msg)
        ordered = tuple(sorted((item for item in symbols if agents[item][1]), key=str))
        return _HeldReductionSpec(
            group,
            reduction.kind,
            tuple(agents[item][1] for item in ordered),
            ordered,
            SR(reduction.argument),
            None if reduction.condition is None else SR(reduction.condition),
        )
