"""Crank-Nicolson for the linear part with a Heun predictor-corrector for the remainder, of second order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core.types import Vector
from ._split import SplitSolver


class CrankNicolsonSolver(SplitSolver):
    """Crank-Nicolson for the linear part with a Heun predictor-corrector for the remainder, of second order.

    The predictor takes a Crank-Nicolson step with the remainder at the start, and the corrector repeats it with the
    remainder averaged over the start and the predicted end. For linear fields it is the trapezoidal rule.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> from skcomplex.engine.solvers import CrankNicolsonSolver
        >>> CrankNicolsonSolver(dt=0.5).dt
        0.5
    """

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one Crank-Nicolson step with a predicted remainder."""
        explicit = x + h / 2 * self.operator_.apply(x)
        start = self._remainder(t, x, p)
        predicted = self.operator_.solve(h / 2, explicit + h * start)
        average = (start + self._remainder(t + h, predicted, p)) / 2
        return self.operator_.solve(h / 2, explicit + h * average)
