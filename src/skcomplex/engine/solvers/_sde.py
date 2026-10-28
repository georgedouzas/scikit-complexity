"""Integrate stochastic differential equations with fixed steps of the Euler-Maruyama, Milstein, and Heun schemes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod

import numpy as np

from ...core.types import Vector
from ..problems import Form
from ._base import BaseSolver
from ._history import History, Interpolant
from ._solver import Solver


class SDESolver(Solver):
    """Base class of fixed-step schemes of stochastic differential equations, which take one increment per step.

    Subclasses set `supports` and implement `_increment`. The steps stop at breakpoints, read delays by linear
    interpolation of the steps, and locate events in them.

    Args:
        dt:
            The largest internal time step.
    """

    @classmethod
    def _get_form(cls) -> Form:
        """Return the form the solver solves."""
        return Form.SDE

    def __init__(
        self,
        dt: float = 1e-3,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.random_state = random_state
        self.until = until

    @abstractmethod
    def _increment(
        self,
        time: float,
        state: Vector,
        p: Vector,
        *,
        step: float | Vector,
        noise: Vector,
        end: float,
    ) -> Vector:
        """Return the state after one step with given Wiener increments, before the boundary laws.

        Args:
            time:
                The time at the start of the step.

            state:
                The state at the start of the step.

            p:
                The parameter vector.

            step:
                The step length, or one step length per state component.

            noise:
                The Wiener increments, one per noise amplitude.

            end:
                The time at the end of the step, capped before the next breakpoint.

        Returns:
            state:
                The state at the end of the step.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _advance_hybrid(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate with fixed steps, reading delays and locating events by linear interpolation of every step."""
        problem = self.problem_
        history = self.history_ = History._open_history(problem, self.history_, t, x)

        def take(time: float, state: Vector, step: float, ceiling: float) -> tuple[Vector, Interpolant]:
            parameters = History._with_delays(problem, history, time, state, p)
            noise = problem.draw_increments(self.rng_, math.sqrt(step))
            following = self._increment(time, state, parameters, step=step, noise=noise, end=min(time + step, ceiling))
            start, end = state.copy(), following.copy()

            def interpolant(s: float) -> Vector:
                return start + (s - time) / step * (end - start)

            return following, interpolant

        return BaseSolver._advance_fixed_steps(
            t,
            x,
            p,
            until,
            dt=self.dt,
            locator=self.locator_,
            history=history,
            bounded=self._bounded,
            take=take,
        )

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate from time `t` to time `until` in fixed steps no longer than `dt`.

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
        if until <= t or self.problem_.drift is None or self.problem_.diffusion is None:
            return x
        return self._advance_hybrid(t, x, p, until)


