"""Parameter symbols of the rewritten elements, in the order of the parameter vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression

from ...core.errors import RewritingError
from .._utils import symbols_of
from ._base import BaseValue
from ._expression_printer import ExpressionPrinter

type Arrays = list[npt.NDArray[Any]]


@dataclass(frozen=True)
class Parameters(BaseValue):
    """Parameter symbols of the rewritten elements, in the order of the parameter vector.

    Args:
        symbols:
            The symbols.

        time:
            The generation symbol.
    """

    symbols: tuple[Expression, ...]
    time: Expression

    def function(
        self,
        variables: Sequence[Expression],
        expressions: Sequence[Any],
        owner: str,
    ) -> Callable[..., Arrays]:
        """Return a NumPy function of the generation, the variables, and the parameters that evaluates expressions.

        Args:
            variables:
                The symbols bound to module parameters.

            expressions:
                The expressions.

            owner:
                The production, named in errors.

        Returns:
            function:
                The function `f(generation, *variables, *parameters)`.

        Raises:
            RewritingError:
                An expression uses a symbol that is neither a variable nor a parameter.
        """
        values = [SR(item) for item in expressions]
        free = symbols_of(values) - set(variables) - set(self.symbols) - {self.time}
        if free:
            error_msg = (
                f'{owner} uses {", ".join(sorted(map(str, free)))}, which is neither a parameter of its predecessor '
                'nor a parameter of its element. Declare it as a `ParameterSpec` argument of the word, patch, or host '
                'graph.'
            )
            raise RewritingError(error_msg)
        return ExpressionPrinter(time=self.time, compiled=True).check().prepare([*variables, *self.symbols]).run(values)
