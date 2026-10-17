"""Define the compiled problem that solvers evaluate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import sympy

from ..core import Form
from ..model import Element
from ._layout import Layout

type Vector = npt.NDArray[np.float64]
type VectorFunction = Callable[[float, Vector, Vector], Vector]


@dataclass(frozen=True)
class CompiledElement:
    """Element of a compiled problem with the dotted names of its variables.

    Args:
        path:
            The slash-separated path of the element in the system tree.

        element:
            The element.

        states:
            A mapping from state name to dotted name.

        parameters:
            A mapping from parameter name to dotted name.
    """

    path: str
    element: Element
    states: Mapping[str, str]
    parameters: Mapping[str, str]


@dataclass(frozen=True)
class Problem:
    """System compiled to one mathematical form with flat state and parameter vectors.

    Only the functions of the problem's form are set. `drift` and `diffusion` belong to the ODE and SDE forms,
    `propensities` and `stoichiometry` to the jump process form, and `update` to the discrete map form.

    Args:
        form:
            The mathematical form.

        states:
            The layout of the state vector.

        parameters:
            The layout of the parameter vector.

        initial_state:
            The initial state vector.

        initial_parameters:
            The initial parameter vector.

        time:
            The time symbol.

        symbols:
            A mapping from dotted name to the symbol of a scalar or the column of symbols of a vector.

        equations:
            A mapping from state component name to its symbolic drift, or to its next value for the discrete form.
            Components with a callable law are left out.

        has_callables:
            Whether any law is a callable.

        elements:
            The compiled elements in state vector order.

        drift:
            The function `f(t, x, p)` that returns the time derivative or the drift.

        diffusion:
            The function `g(t, x, p)` that returns one column per independent Wiener process.

        propensities:
            The function `a(t, x, p)` that returns one rate per jump.

        stoichiometry:
            The increments of the state vector, one row per jump.

        update:
            The function `F(t, x, p)` that returns the next state vector.
    """

    form: Form
    states: Layout
    parameters: Layout
    initial_state: Vector
    initial_parameters: Vector
    time: sympy.Symbol
    symbols: Mapping[str, sympy.Symbol | sympy.MatrixBase]
    equations: Mapping[str, sympy.Expr]
    has_callables: bool
    elements: tuple[CompiledElement, ...]
    drift: VectorFunction | None = field(default=None, repr=False)
    diffusion: VectorFunction | None = field(default=None, repr=False)
    propensities: VectorFunction | None = field(default=None, repr=False)
    stoichiometry: Vector | None = field(default=None, repr=False)
    update: VectorFunction | None = field(default=None, repr=False)

    def _flatten_symbols(self, layout: Layout) -> list[sympy.Symbol]:
        symbols: list[sympy.Symbol] = []
        for entry in layout:
            symbol = self.symbols[entry.name]
            symbols.extend(symbol if isinstance(symbol, sympy.MatrixBase) else [symbol])
        return symbols

    @property
    def state_symbols(self) -> list[sympy.Symbol]:
        """Symbols of the state components, in vector order."""
        return self._flatten_symbols(self.states)

    @property
    def parameter_symbols(self) -> list[sympy.Symbol]:
        """Symbols of the parameter components, in vector order."""
        return self._flatten_symbols(self.parameters)
