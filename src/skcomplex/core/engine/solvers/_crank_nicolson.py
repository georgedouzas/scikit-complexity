"""Crank-Nicolson for the linear part with a Heun predictor-corrector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import Array
from ._split import SplitSolver


class CrankNicolsonSolver(SplitSolver):
    """A solver that steps the linear part by Crank-Nicolson and the rest by Heun.

    A predictor takes a Crank-Nicolson step, the trapezoidal rule for the linear part, with the rest at the start of the
    step, and a corrector repeats it with the rest averaged over the start and the predicted end, so the scheme is of
    order 2. For a linear field, such as the heat of a plate, it is the trapezoidal rule. Choose it for stiff diffusion
    with a mild nonlinear part, in models of heat flow, reaction and diffusion, and option prices.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The fixed internal step.

    See Also:
        IMEX2Solver: Guesses the rest from the last two steps.

    Examples:
        >>> from skcomplex.core.engine.solvers import CrankNicolsonSolver
        >>> from skcomplex.systems.physics.heat import HeatDiffusionSystem
        >>> solver = CrankNicolsonSolver(dt=0.5)
        >>> u = HeatDiffusionSystem((16, 16), solver=solver).prepare().simulate(until=2).get_values('plate.u')
        >>> round(float(u[0].sum()), 6), round(float(u[-1].sum()), 6)
        (16.0, 16.0)
    """

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one Crank-Nicolson step with a predicted remainder."""
        explicit = x + h / 2 * self._get_linear(x)
        start = self._get_remainder(t, x, p)
        predicted = self._solve_linear(h / 2, explicit + h * start)
        average = (start + self._get_remainder(t + h, predicted, p)) / 2
        return self._solve_linear(h / 2, explicit + h * average)
