"""Explicit Grünwald-Letnikov scheme of Caputo fractional differential equations, of order 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...types import Array
from ._convolution import _MODE_SPACING, ConvolutionSolver


class GrunwaldLetnikovSolver(ConvolutionSolver):
    """Explicit Grünwald-Letnikov scheme of Caputo fractional differential equations, of order 1.

    A step sets `dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^k w_j (x_(k-j) - x_0)` with the weights
    `w_j = (-1)^j binomial(alpha, j)`. States of order 1 take explicit Euler steps.

    Args:
        dt:
            The fixed step. Frames must fall on multiples of it.

        memory:
            The length of time over which the history enters with its exact weights, or `None` for the whole run. The
            history older than the window enters through exponential modes fitted to the weights, so a step costs
            the same however long the run.

    Attributes:
        orders_:
            The order of every state component.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.specs import StateSpec
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.engine.solvers import GrunwaldLetnikovSolver, Step
        >>> class RelaxingElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec(order=0.5)] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, s):
        ...         return [RateLaw(s.x, -s.x)]
        >>> problem = compile_system(System(elements=[RelaxingElement('a')]))
        >>> stepper = GrunwaldLetnikovSolver(dt=0.001, random_state=0).check().prepare(problem)
        >>> round(float(stepper.run(Step(0.0, problem.initial_state, problem.initial_parameters, 1.0))[0]), 2)
        0.43
    """

    def _weights(self, alpha: Array, count: int) -> Array:
        """Return the Grünwald-Letnikov weights `w_0` to `w_count`, one column per order."""
        return self._grunwald_letnikov_weights(alpha, count)

    def _tail(self, alpha: Array, rates: Array, window: int) -> Array:
        """Return the coefficients of the exponential modes of the Grünwald-Letnikov weights past a window."""
        return self._grunwald_letnikov_tail(alpha, rates, window)

    def _gain(self, alpha: Array) -> Array:
        """Return the factor 1 of the drift of every order."""
        return np.ones_like(alpha)

    @staticmethod
    def _grunwald_letnikov_weights(alpha: Array, count: int) -> Array:
        """Return the Grünwald-Letnikov weights `w_0` to `w_count`, one column per order."""
        lags = np.arange(1, count + 1, dtype=float)[:, None]
        factors = 1 - (alpha[None, :] + 1) / lags
        return np.vstack([np.ones((1, alpha.size)), np.cumprod(factors, axis=0)])

    @staticmethod
    def _grunwald_letnikov_tail(alpha: Array, rates: Array, window: int) -> Array:
        """Return the coefficients of the exponential modes whose sum gives the Grünwald-Letnikov weights past a window.

        The weights have the Laplace representation `w_j = -(sin(pi alpha) / pi) int exp(-s j) exp(s alpha)
        (1 - exp(-s))^alpha ds`, discretised with the trapezoidal rule in `log s`.
        """
        s, a = rates[:, None], alpha[None, :]
        density = -np.sin(np.pi * a) / np.pi * np.exp(-s * (window + 1 - a)) * (-np.expm1(-s)) ** a
        return _MODE_SPACING * s * density
