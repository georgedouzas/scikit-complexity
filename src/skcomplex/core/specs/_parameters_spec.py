"""Define the parameters spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ParametersSpec(BaseSpec):
    """Parameter symbols of the rewritten elements.

    They are in the order of the parameter vector.

    Args:
        symbols:
            The symbols.

        time:
            The generation symbol.
    """

    symbols: tuple[Expression, ...]
    time: Expression
