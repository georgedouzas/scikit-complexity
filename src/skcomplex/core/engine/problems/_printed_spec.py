"""Define the printed spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec


@dataclass(frozen=True)
class PrintedSpec(BaseSpec):
    """A spec of the printed code of an expression and how tightly it binds.

    When the engine prints a law as code, it keeps with every piece of code the precedence of its outermost operation,
    so that an operator around it adds parentheses only when needed, such as `(x + 1) * y` but `x * y + 1`. It prints
    the rates of a population, the forces of a pendulum, and the prices of a market.

    Read more in the user guide, under Backends.

    Args:
        code:
            The code of the expression.

        precedence:
            The precedence of the outermost operation of the code, higher for tighter binding.

    See Also:
        ExpressionPrinter: Prints expressions as code.

    Examples:
        >>> from skcomplex.core.engine.problems._printed_spec import PrintedSpec
        >>> PrintedSpec('x + 1', 10).code
        'x + 1'
    """

    code: str
    precedence: int
