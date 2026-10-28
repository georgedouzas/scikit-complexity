"""Implicit-explicit Euler: the linear part implicit and the nonlinear remainder explicit, of first order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core.types import Vector
from ._split import _SplitSolver


class IMEXEulerSolver(_SplitSolver):
    """Implicit-explicit Euler: the linear part implicit and the nonlinear remainder explicit, of first order.

    Every step solves `(I - h L) x_{n+1} = x_n + h N(x_n)`, with sparse solves for finite differences and finite
    elements and division per Fourier mode for spectral fields.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.physics.heat import heat_diffusion
        >>> from skcomplex.engine.solvers import IMEXEulerSolver
        >>> problem = compile_system(heat_diffusion((16, 16)))
        >>> stepper = IMEXEulerSolver(dt=1.0).prepare(problem, np.random.default_rng(0))
        >>> state = stepper.run(0.0, problem.initial_state, problem.initial_parameters, 10.0)
        >>> bool(np.isclose(state.sum(), problem.initial_state.sum()))
        True
    """

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one implicit-explicit Euler step."""
        return self.operator_.solve(h, x + h * self._remainder(t, x, p))
