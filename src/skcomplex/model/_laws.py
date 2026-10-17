"""Define the law kinds that describe how state changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import sympy

type Target = sympy.Symbol | sympy.MatrixBase
type NumericFunction = Callable[[float, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Rate:
    """Contribution to the time derivative of a state.

    Rates on the same state add up, so a system with only rate laws is a system of ordinary differential equations.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression, anything SymPy can convert to one, or a callable `f(t, values)` where `values`
            maps dotted state and parameter names to their current values.

    Examples:
        >>> import sympy
        >>> from skcomplex.model import Rate
        >>> n = sympy.Symbol('n')
        >>> Rate(n, 2 * n).value
        2*n
    """

    target: Target
    value: object


@dataclass(frozen=True)
class Noise:
    """Amplitude of an independent Wiener process that drives a state.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)`. A vector target gets one Wiener process per component.

    Examples:
        >>> import sympy
        >>> from skcomplex.model import Noise
        >>> price = sympy.Symbol('price')
        >>> Noise(price, 0.2 * price).value
        0.2*price
    """

    target: Target
    value: object


@dataclass(frozen=True)
class Jump:
    """Random event that changes states by fixed increments at a given rate.

    Args:
        rate:
            The propensity of the event, as a symbolic expression or a callable `f(t, values)`.

        changes:
            A mapping from the symbol of a scalar state to its increment when the event fires.

    Examples:
        >>> import sympy
        >>> from skcomplex.model import Jump
        >>> s, i = sympy.symbols('s i')
        >>> Jump(0.1 * s * i, {s: -1, i: 1}).changes[i]
        1
    """

    rate: object
    changes: Mapping[sympy.Symbol, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Update:
    """Next value of a state after one discrete time step.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression or a callable `f(t, values)` that gives the next value.

    Examples:
        >>> import sympy
        >>> from skcomplex.model import Update
        >>> x = sympy.Symbol('x')
        >>> Update(x, 3.7 * x * (1 - x)).target
        x
    """

    target: Target
    value: object


type Law = Rate | Noise | Jump | Update
