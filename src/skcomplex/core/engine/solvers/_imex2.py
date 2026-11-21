"""Crank-Nicolson Adams-Bashforth scheme of second order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import Array
from ._split import SplitSolver


class IMEX2Solver(SplitSolver):
    """A solver of order 2 that steps stiff parts implicitly and the rest explicitly.

    Every step solves `(I - h L / 2) x_{n+1} = (I + h L / 2) x_n + h (3 N_n - N_{n-1}) / 2`, the trapezoidal rule for
    the linear part `L` and a guess of the rest `N` from the last two steps. It starts with one Crank-Nicolson step, and
    a call that does not go on from the time and state of the last call starts again. Choose it for fields with stiff
    diffusion and reactions, such as the Gray-Scott patterns of chemistry, the pulses of a nerve, and the heat of a
    plate.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The fixed internal step.

    See Also:
        IMEXEulerSolver: The same split with order 1.

    Examples:
        >>> from skcomplex.core.engine.solvers import IMEX2Solver
        >>> from skcomplex.systems.physics.heat import HeatDiffusionSystem
        >>> system = HeatDiffusionSystem((16, 16), solver=IMEX2Solver(dt=0.5)).prepare()
        >>> u = system.simulate(until=2).get_values('plate.u')
        >>> round(float(u[0].sum()), 6), round(float(u[-1].sum()), 6)
        (16.0, 16.0)
    """

    def _forget(self) -> None:
        """Forget the previous remainder.

        The next step then starts the scheme again.
        """
        self._previous: Array | None = None

    def _check_built(self, p: Array) -> None:
        """Forget the previous remainder when the linear part changes."""
        self._forget()

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one Crank-Nicolson Adams-Bashforth step."""
        remainder = self._get_remainder(t, x, p)
        extrapolated = remainder if self._previous is None else 1.5 * remainder - 0.5 * self._previous
        self._previous = remainder
        return self._solve_linear(h / 2, x + h / 2 * self._get_linear(x) + h * extrapolated)
