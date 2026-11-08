"""Crank-Nicolson Adams-Bashforth scheme of second order.

It has a trapezoidal linear part and an extrapolated remainder.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import Array
from ._split import SplitSolver


class IMEX2Solver(SplitSolver):
    """Crank-Nicolson Adams-Bashforth scheme of second order.

    It has a trapezoidal linear part and an extrapolated remainder. Every step solves `(I - h L / 2) x_{n+1} = (I + h L
    / 2) x_n + h (3 N_n - N_{n-1}) / 2`, starting with one Crank-Nicolson step with an Euler remainder. A call that does
    not continue the time, state, and step of the previous call starts the scheme again.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> from skcomplex.core.engine.solvers import IMEX2Solver
        >>> IMEX2Solver(dt=0.1).dt
        0.1
    """

    def _forget(self) -> None:
        """Forget the previous remainder.

        The next step then starts the scheme again.
        """
        self._previous: Array | None = None

    def _built(self, p: Array) -> None:
        """Forget the previous remainder when the linear part changes."""
        self._forget()

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one Crank-Nicolson Adams-Bashforth step."""
        remainder = self._remainder(t, x, p)
        extrapolated = remainder if self._previous is None else 1.5 * remainder - 0.5 * self._previous
        self._previous = remainder
        return self._solve_linear(h / 2, x + h / 2 * self._linear(x) + h * extrapolated)
