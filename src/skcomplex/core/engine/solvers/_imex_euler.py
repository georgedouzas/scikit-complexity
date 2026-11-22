"""Implicit-explicit Euler of first order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import Array
from ._split import SplitSolver


class IMEXEulerSolver(SplitSolver):
    """A solver of order 1 that steps stiff parts implicitly and the rest explicitly.

    Every step solves `(I - h L) x_{n+1} = x_n + h N(x_n)` for the linear part `L`, such as diffusion, and the rest `N`,
    such as a reaction, so a large step stays stable where an explicit step would blow up. It solves sparse systems for
    finite differences and finite elements and divides per Fourier mode for spectral fields. Choose it as a stable first
    try on a stiff field, in models of heat flow, patterns, and the spread of a species.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The fixed internal step.

    See Also:
        IMEX2Solver: The same split with order 2.
        CrankNicolsonSolver: Steps the linear part by the trapezoidal rule.

    Examples:
        >>> from skcomplex.core.engine.solvers import IMEXEulerSolver
        >>> from skcomplex.systems.physics.heat import HeatDiffusionSystem
        >>> system = HeatDiffusionSystem((16, 16), solver=IMEXEulerSolver(dt=0.5)).prepare()
        >>> u = system.simulate(until=2).get_values('plate.u')
        >>> round(float(u[0].sum()), 6), round(float(u[-1].sum()), 6)
        (16.0, 16.0)
    """

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one implicit-explicit Euler step."""
        return self._solve_linear(h, x + h * self._get_remainder(t, x, p))
