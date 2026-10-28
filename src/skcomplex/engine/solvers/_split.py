"""Base of steppers that treat the linear part of the rates of fields apart from the nonlinear remainder."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np

from ...core.errors import SolverError
from ...core.laws import BoundaryLaw, FluxLaw, RateLaw, TransferLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...core.types import Vector
from ..problems import Form, OrderKind, Problem, SupportSpec, step_count
from ._solver import Solver
from ._split_operator import SplitOperator


class _SplitSolver(Solver):
    """Base of steppers that treat the linear part of the rates of fields apart from the nonlinear remainder."""

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={RateLaw, FluxLaw, BoundaryLaw, TransferLaw},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.ODE
    implicit: ClassVar[bool] = True

    def __init__(self, dt: float = 0.01) -> None:
        self.dt = dt

    def _prepare(self, problem: Problem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the stepper and check that the problem has linear terms to treat apart.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator, unused.

            until:
                The end time of the run.

        Raises:
            SolverError:
                The problem is not an ODE, has no linear terms of fields, or has delays or events.
        """
        super()._prepare(problem, rng, until)
        if not problem.linear:
            error_msg = (
                f'{type(self).__name__} splits the rates of fields into linear and nonlinear parts, but the problem '
                'has no linear terms with constant coefficients. Use ScipyODESolver or RK4Solver.'
            )
            raise SolverError(error_msg)
        if problem.delays or problem.events:
            error_msg = f'{type(self).__name__} does not integrate delays or events. Use ScipyODESolver or RK4Solver.'
            raise SolverError(error_msg)
        self._parameters: Vector | None = None
        self._last: tuple[float, Vector, float] | None = None

    def _continues(self, t: float, x: Vector, h: float) -> bool:
        """Check whether a call continues the time, state, and step of the previous call."""
        if self._last is None:
            return False
        time, state, step = self._last
        tolerance = 1e-9 * max(1.0, abs(t))
        return abs(t - time) <= tolerance and abs(h - step) <= tolerance and np.array_equal(x, state)

    def _forget(self) -> None:
        """Hook that forgets the steps of earlier calls, which multistep schemes override."""

    def _build_operator(self, p: Vector) -> None:
        """Build the linear part for parameter values when they change."""
        if self._parameters is None or not np.array_equal(p, self._parameters):
            self.operator_ = SplitOperator.build(self.problem_, p)
            self._parameters = p.copy()
            self._built(p)

    def _built(self, p: Vector) -> None:
        """Hook that runs after the linear part is rebuilt."""

    def _remainder(self, t: float, x: Vector, p: Vector) -> Vector:
        """Return the nonlinear remainder of the rates, the rates minus their linear part."""
        return self.problem_.drift(t, x, p) - self.operator_.apply(x)  # type: ignore[misc]

    def _step(self, t: float, x: Vector, p: Vector, h: float) -> Vector:
        """Return the state after one step."""
        raise NotImplementedError

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
