"""Define the printed spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...specs._base_spec import BaseSpec


@dataclass(frozen=True)
class PrintedSpec(BaseSpec):
    """Code of an expression and how tightly it binds.

    How tightly it binds decides whether an operator around it needs parentheses.

    Args:
        code:
            The code of the expression.

        precedence:
            The precedence of the outermost operation of the code, higher for tighter binding.
    """

    code: str
    precedence: int
