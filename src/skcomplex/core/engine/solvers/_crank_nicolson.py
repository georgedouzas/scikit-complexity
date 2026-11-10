"""Crank-Nicolson for the linear part with a Heun predictor-corrector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import Array
from ._split import SplitSolver


class CrankNicolsonSolver(SplitSolver):
    """A solver that steps the linear part by Crank-Nicolson, the rest by Heun.

    The predictor-corrector handles the remainder, and the scheme is of second order. The predictor takes a
    Crank-Nicolson step with the remainder at the start, and the corrector repeats it with the remainder averaged over
    the start and the predicted end. For linear fields it is the trapezoidal rule.

    Choose it for stiff diffusion with a mild nonlinear part. It appears in models of heat flow, reaction diffusion, and
    option pricing.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> from skcomplex.core.engine.solvers import CrankNicolsonSolver
        >>> CrankNicolsonSolver(dt=0.5).dt
        0.5
    """

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one Crank-Nicolson step with a predicted remainder."""
        explicit = x + h / 2 * self._get_linear(x)
        start = self._get_remainder(t, x, p)
        predicted = self._solve_linear(h / 2, explicit + h * start)
        average = (start + self._get_remainder(t + h, predicted, p)) / 2
        return self._solve_linear(h / 2, explicit + h * average)
