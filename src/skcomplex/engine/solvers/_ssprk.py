"""Strong-stability-preserving Runge-Kutta of order 2 or 3, with steps from the CFL condition of conservation laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...core.laws import BoundaryLaw, FluxLaw, RateLaw, TransferLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems import ProblemSpec
from ._solver import Solver


class SSPRKSolver(Solver):
    """Strong-stability-preserving Runge-Kutta of order 2 or 3, with steps from the CFL condition of conservation laws.

    The schemes of Shu and Osher are convex combinations of Euler steps, so they keep the total variation bound of a
    limited finite-volume scheme. With finite-volume fields the step is the smaller of `cfl * h / s`, for the cell size
    `h` and the largest wave speed `s`, and the stability limit of the Laplacian rates at the current parameters, and
    otherwise `dt`.

    Args:
        order:
            2 or 3.

        cfl:
            The Courant number.

        dt:
            The largest step, and the step without conservation laws.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.physics.fluids import burgers
        >>> from skcomplex.engine.solvers import SSPRKSolver
        >>> from skcomplex.specs import StepSpec
        >>> problem = compile_system(burgers(64))
        >>> stepper = SSPRKSolver(random_state=0).check().prepare(problem)
        >>> state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))
        >>> bool(np.isclose(state.sum(), problem.initial_state.sum()))
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={RateLaw, FluxLaw, BoundaryLaw, TransferLaw},
            orders={OrderKind.INTEGER},
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

    def __init__(
        self,
        order: int = 3,
        cfl: float = 0.4,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.order = order
        self.cfl = cfl
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _prepare(self, problem: ProblemSpec) -> None:
        """Prepare the stepper and check its order.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The order is not 2 or 3, or the problem is not an ODE.
        """
        super()._prepare(problem)
        if self.order not in {2, 3}:
            error_msg = f'SSPRKSolver has order 2 or 3, not {self.order}.'
            raise SolverError(error_msg)
        self.spacing_ = min((min(volume.spacings) for volume in problem.volumes), default=None)

    def _step_size(self, x: Array, p: Array) -> float:
        """Return the step allowed by the CFL condition and the diffusion of finite-volume fields, at most `dt`."""
        problem = self.problem_
        step = min([self.dt, *(limit.step(p) for limit in problem.diffusive_limits)])
        speed = max((volume.max_speed(x, p) for volume in problem.volumes), default=None)
        if speed is None or self.spacing_ is None or speed <= 0:
            return step
        return min(step, self.cfl * self.spacing_ / speed)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate from time `t` to time `until` with steps of the CFL condition.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector at time `until`.
        """
        drift = self.problem_.drift
        state, time = x.copy(), t
        while drift is not None and time < until - 1e-12 * max(1.0, abs(until)):
            h = min(self._step_size(state, p), until - time)
            first = state + h * drift(time, state, p)
            if self.order == 2:  # noqa: PLR2004
                state = (state + first + h * drift(time + h, first, p)) / 2
            else:
                second = 0.75 * state + 0.25 * (first + h * drift(time + h, first, p))
                state = state / 3 + 2 / 3 * (second + h * drift(time + h / 2, second, p))
            time += h
            state = self._bounded(time, state, p)
        return state
