"""Helpers shared by the controllers: clipping a control input and reading a variable of a plant."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.laws.utils import where


def _limited(value: Expression, limits: tuple[float, float] | None) -> Expression:
    """Return a value clipped to limits, or the value itself without limits."""
    if limits is None:
        return value
    lower, upper = limits
    return where(value > upper, upper, where(value < lower, lower, value))


def _read(plant: SimpleNamespace, name: str) -> Expression:
    """Return the symbol of a variable of a plant by its name, such as `x` or its velocity `x_d1`."""
    value: Expression = getattr(plant, name)
    return value
