"""Integrate stochastic differential equations with fixed steps of the Euler-Maruyama, Milstein, and Heun schemes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod

import numpy as np

from ...types import Array, FormKind
from ..types import Interpolant
from ._base import BaseSolver
from ._history import History
from ._solver import Solver


class SDESolver(Solver):
    """Base class of fixed-step schemes of stochastic differential equations, which take one increment per step.

    Subclasses set `supports` and implement `_increment`. The steps stop at breakpoints, read delays by linear
    interpolation of the steps, and locate events in them.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.
    """

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.SDE

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
        state: Array,
        p: Array,
        *,
        step: float | Array,
        noise: Array,
        end: float,
    ) -> Array:
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

    def _draw_increments(self, scale: float) -> Array:
        """Return the Wiener increment that every noise amplitude of the problem multiplies, drawn for one step.

        The increments of the processes are normal with the standard deviation `scale`, correlated within every block
        by its covariance factor. Amplitudes that share a process take its increment.

        Args:
            scale:
                The square root of the step.

        Returns:
            increments:
                One increment per noise amplitude, in the order of `noise_rows`.
        """
        problem = self.problem_
        increments = self.rng_.normal(0.0, scale, size=problem.n_processes)
        for block in problem.noise_blocks:
            part = slice(block.start, block.start + block.size)
            increments[part] = block.factor @ increments[part]
        return increments[problem.noise_columns]

    def _advance_hybrid(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate with fixed steps, reading delays and locating events by linear interpolation of every step."""
        problem = self.problem_
        history = self.history_ = History._open_history(problem, self.history_, t, x)

        def take(time: float, state: Array, step: float, ceiling: float) -> tuple[Array, Interpolant]:
            parameters = History._with_delays(problem, history, time, state, p)
            noise = self._draw_increments(math.sqrt(step))
            following = self._increment(time, state, parameters, step=step, noise=noise, end=min(time + step, ceiling))
            start, end = state.copy(), following.copy()

            def interpolant(s: float) -> Array:
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

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
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
