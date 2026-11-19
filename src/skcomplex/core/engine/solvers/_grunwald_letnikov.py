"""Explicit Grünwald-Letnikov scheme of Caputo fractional differential equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...types import Array
from ._convolution import MODE_SPACING, ConvolutionSolver


class GrunwaldLetnikovSolver(ConvolutionSolver):
    """A solver of Caputo fractional equations by the Grünwald-Letnikov scheme.

    It is of order 1. A step sets `dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^k w_j (x_(k-j) - x_0)` with the weights `w_j
    = (-1)^j binomial(alpha, j)`. States of order 1 take explicit Euler steps.

    Choose it for a simple explicit fractional scheme with a small step. It appears in models of anomalous diffusion and
    of fractional chaotic flows.

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
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.interactions import Interaction
        >>> from skcomplex.core.components.laws import RateLaw
        >>> from skcomplex.core.components.laws.utils import fractional
        >>> from skcomplex.core.engine.solvers import GrunwaldLetnikovSolver, StepSpec
        >>> from skcomplex.core.systems import System
        >>> def relax(s):
        ...     return [RateLaw(fractional(s.target.x, 0.5), -s.target.x)]
        >>> relaxation = Interaction('relaxation', roles={'target': 'a'}, laws=relax)
        >>> system = System(elements=[Element('a', values={'x': 1.0})], interactions=[relaxation])
        >>> problem = system.prepare().simulate(until=0).problem
        >>> stepper = GrunwaldLetnikovSolver(dt=0.001, random_state=0).prepare(problem)
        >>> step = StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> round(float(stepper.run(step).state_[0]), 2)
        0.43
    """

    def _get_weights(self, alpha: Array, count: int) -> Array:
        """Return the Grünwald-Letnikov weights `w_0` to `w_count`, one column per order."""
        return self._get_grunwald_letnikov_weights(alpha, count)

    def _get_tail(self, alpha: Array, rates: Array, window: int) -> Array:
        """Return the coefficients of the exponential modes of the tail weights.

        They are the modes of the Grünwald-Letnikov weights past a window.
        """
        return self._get_grunwald_letnikov_tail(alpha, rates, window)

    def _get_gain(self, alpha: Array) -> Array:
        """Return the factor 1 of the drift of every order."""
        return np.ones_like(alpha)

    @staticmethod
    def _get_grunwald_letnikov_weights(alpha: Array, count: int) -> Array:
        """Return the Grünwald-Letnikov weights `w_0` to `w_count`, one column per order."""
        lags = np.arange(1, count + 1, dtype=float)[:, None]
        factors = 1 - (alpha[None, :] + 1) / lags
        return np.vstack([np.ones((1, alpha.size)), np.cumprod(factors, axis=0)])

    @staticmethod
    def _get_grunwald_letnikov_tail(alpha: Array, rates: Array, window: int) -> Array:
        """Return the coefficients of the exponential modes of the tail weights.

        The sum of the modes gives the Grünwald-Letnikov weights past a window. The weights have the Laplace
        representation `w_j = -(sin(pi alpha) / pi) int exp(-s j) exp(s alpha) (1 - exp(-s))^alpha ds`, discretised with
        the trapezoidal rule in `log s`.
        """
        s, a = rates[:, None], alpha[None, :]
        density = -np.sin(np.pi * a) / np.pi * np.exp(-s * (window + 1 - a)) * (-np.expm1(-s)) ** a
        return MODE_SPACING * s * density
