"""Implicit-explicit Euler of first order.

The linear part is implicit and the nonlinear remainder explicit.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import Array
from ._split import SplitSolver


class IMEXEulerSolver(SplitSolver):
    """Implicit-explicit Euler of first order.

    The linear part is implicit and the nonlinear remainder explicit. Every step solves `(I - h L) x_{n+1} = x_n + h
    N(x_n)`, with sparse solves for finite differences and finite elements and division per Fourier mode for spectral
    fields.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.physics.heat import HeatDiffusionSystem
        >>> from skcomplex.core.engine.solvers import IMEXEulerSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> problem = simulate(HeatDiffusionSystem((16, 16)), until=0).problem
        >>> stepper = IMEXEulerSolver(dt=1.0, random_state=0).prepare(problem)
        >>> state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 10.0)).state_
        >>> bool(np.isclose(state.sum(), problem.initial_state.sum()))
        True
    """

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one implicit-explicit Euler step."""
        return self._solve_linear(h, x + h * self._get_remainder(t, x, p))
