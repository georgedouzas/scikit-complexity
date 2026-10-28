"""Base of steppers that treat the linear part of the rates of fields apart from the nonlinear remainder."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ...core.laws import BoundaryLaw, FluxLaw, RateLaw, TransferLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import ProblemSpec, SupportSpec
from ...types import Array, FormKind, OrderKind
from ..problems._steps import step_count
from ._solver import Solver
from ._split_operator import SplitOperator


class SplitSolver(Solver):
    """Base of steppers that treat the linear part of the rates of fields apart from the nonlinear remainder."""

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

    @classmethod
    def _is_implicit(cls) -> bool:
        """Return whether the solver treats diffusion implicitly, so explicit stability limits do not apply."""
        return True

    def __init__(
        self,
        dt: float = 0.01,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.random_state = random_state
        self.until = until

    def _prepare(self, problem: ProblemSpec) -> None:
        """Prepare the stepper and check that the problem has linear terms to treat apart.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not an ODE, has no linear terms of fields, or has delays or events.
        """
        super()._prepare(problem)
        if not problem.linear:
            error_msg = (
                f'{type(self).__name__} splits the rates of fields into linear and nonlinear parts, but the problem '
                'has no linear terms with constant coefficients. Use ScipyODESolver or RK4Solver.'
            )
            raise SolverError(error_msg)
        if problem.delays or problem.events:
            error_msg = f'{type(self).__name__} does not integrate delays or events. Use ScipyODESolver or RK4Solver.'
            raise SolverError(error_msg)
        self._parameters: Array | None = None
        self._last: tuple[float, Array, float] | None = None

    def _continues(self, t: float, x: Array, h: float) -> bool:
        """Check whether a call continues the time, state, and step of the previous call."""
        if self._last is None:
            return False
        time, state, step = self._last
        tolerance = 1e-9 * max(1.0, abs(t))
        return abs(t - time) <= tolerance and abs(h - step) <= tolerance and np.array_equal(x, state)

    def _forget(self) -> None:
        """Hook that forgets the steps of earlier calls, which multistep schemes override."""

    def _build_operator(self, p: Array) -> None:
        """Build the linear part for parameter values when they change."""
        if self._parameters is None or not np.array_equal(p, self._parameters):
            self.operator_ = SplitOperator.build(self.problem_, p)
            self._parameters = p.copy()
            self._built(p)

    def _built(self, p: Array) -> None:
        """Hook that runs after the linear part is rebuilt."""

    def _remainder(self, t: float, x: Array, p: Array) -> Array:
        """Return the nonlinear remainder of the rates, the rates minus their linear part."""
        return self.problem_.drift(t, x, p) - self.operator_.apply(x)  # type: ignore[misc]

    def _step(self, t: float, x: Array, p: Array, h: float) -> Array:
        """Return the state after one step."""
        raise NotImplementedError

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate from time `t` to time `until` in equal steps no longer than `dt`.

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
        if until <= t:
            return x
        self._build_operator(p)
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        if not self._continues(t, x, step):
            self._forget()
        state = x.copy()
        for index in range(n_steps):
            time = t + (index + 1) * step
            following = self._step(t + index * step, state, p, step)
            state = self._bounded(time, following, p)
            if not np.array_equal(state, following):
                self._forget()
        self._last = (until, state.copy(), step)
        return state
