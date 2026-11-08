"""Explicit L1 scheme of Caputo fractional differential equations, of order 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from scipy.special import gamma

from ...types import Array
from ._convolution import _MODE_SPACING, ConvolutionSolver


class L1Solver(ConvolutionSolver):
    """Explicit L1 scheme of Caputo fractional differential equations, of order 1.

    The L1 scheme interpolates the state linearly between grid points inside the Caputo integral. A step sets
    `Gamma(2 - alpha) dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^(k-1) b_j (x_(k-j) - x_(k-j-1))` with the weights
    `b_j = (j + 1)^(1 - alpha) - j^(1 - alpha)`. States of order 1 take explicit Euler steps.

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
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.solvers import L1Solver
        >>> from skcomplex.core.specs import StepSpec
        >>> class RelaxingElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec(order=0.5)] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, s):
        ...         return [RateLaw(s.x, -s.x)]
        >>> problem = simulate(System(elements=[RelaxingElement('a')]), until=0).problem
        >>> stepper = L1Solver(dt=0.001, random_state=0).prepare(problem)
        >>> step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> round(float(stepper.run(step).state_[0]), 2)
        0.43
    """

    def _weights(self, alpha: Array, count: int) -> Array:
        """Return the weights `a_0` to `a_count` of the L1 scheme on the displacements, one column per order."""
        return self._l1_weights(alpha, count)

    def _tail(self, alpha: Array, rates: Array, window: int) -> Array:
        """Return the coefficients of the exponential modes of the L1 weights past a window."""
        return self._l1_tail(alpha, rates, window)

    def _gain(self, alpha: Array) -> Array:
        """Return the factor `Gamma(2 - alpha)` of the drift of every order."""
        return np.asarray(gamma(2 - alpha))

    @staticmethod
    def _l1_weights(alpha: Array, count: int) -> Array:
        """Return the weights `a_0` to `a_count` of the L1 scheme on displacements from the start, one column per order.

        The weights are the differences `b_j - b_(j-1)` of the L1 weights `b_j = (j + 1)^(1 - alpha) - j^(1 - alpha)` on
        the increments, with `b_0 = 1`, so the sums run over the displacements from the initial state.
        """
        lags = np.arange(count + 1, dtype=float)[:, None]
        exponent = 1 - alpha[None, :]
        increments = (lags + 1) ** exponent - lags**exponent
        increments[0] = 1.0
        return np.vstack([np.ones((1, alpha.size)), np.diff(increments, axis=0)])

    @staticmethod
    def _l1_tail(alpha: Array, rates: Array, window: int) -> Array:
        """Return the coefficients of the exponential modes whose sum gives the L1 weights past a window.

        The weights have the Laplace representation `a_j = -((1 - alpha) / Gamma(alpha)) int exp(-s j) s^(alpha - 2)
        (1 - exp(-s))^2 exp(s) ds`, discretised with the trapezoidal rule in `log s`.
        """
        s, a = rates[:, None], alpha[None, :]
        density = -(1 - a) / gamma(a) * s ** (a - 2) * np.expm1(-s) ** 2 * np.exp(-s * window)
        return _MODE_SPACING * s * density
