"""Integrate stochastic differential equations with fixed steps of the Euler-Maruyama, Milstein, and Heun schemes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod
from typing import ClassVar, cast

import numpy as np

from ...core.base import (
    NUMERIC_DOMAINS,
    ElementKind,
    Form,
    LawKind,
    NoiseKind,
    OrderKind,
    SolverError,
    SpaceKind,
    Supports,
    TimeKind,
    Vector,
)
from ...core.problems import VectorFunction
from ._hybrid import Interpolant, advance_fixed_steps, is_hybrid, open_history, with_delays
from ._stepper import Solver, register_solver, step_count


class SDESolver(Solver):
    """Base class of fixed-step schemes of stochastic differential equations, which take one increment per step.

    Subclasses set `supports` and implement `_increment`. The steps stop at breakpoints, read delays by linear
    interpolation of the steps, and locate events in them.

    Args:
        dt:
            The largest internal time step.
    """

    form: ClassVar[Form] = Form.SDE

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

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
        history = self.history_ = open_history(problem, self.history_, t, x)

        def take(time: float, state: Vector, step: float, ceiling: float) -> tuple[Vector, Interpolant]:
            parameters = with_delays(problem, history, time, state, p)
            noise = problem.draw_increments(self.rng_, math.sqrt(step))
            following = self._increment(time, state, parameters, step=step, noise=noise, end=min(time + step, ceiling))
            start, end = state.copy(), following.copy()

            def interpolant(s: float) -> Vector:
                return start + (s - time) / step * (end - start)

            return following, interpolant

        return advance_fixed_steps(
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


class EulerMaruyamaSolver(SDESolver):
    """Euler-Maruyama integrator of stochastic differential equations with a fixed internal step.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.economics.finance import AssetElement, GeometricBrownianMotionInteraction
        >>> from skcomplex.engine.solvers import EulerMaruyamaSolver
        >>> problem = compile_system(
        ...     System(
        ...         components=[AssetElement('stock', price=100)],
        ...         interactions=[GeometricBrownianMotionInteraction('gbm', target='stock', mu=0.05, sigma=0.2)],
        ...     ),
        ... )
        >>> stepper = EulerMaruyamaSolver(dt=1e-3).prepare(problem, np.random.default_rng(0))
        >>> bool(stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0] > 0)
        True
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        space=set(SpaceKind.registered()),
        laws={
            LawKind.JUMP,
            LawKind.RATE,
            LawKind.EQUATION,
            LawKind.EVENT,
            LawKind.BOUNDARY,
            LawKind.TRANSFER,
            LawKind.NOISE,
        },
        domains=NUMERIC_DOMAINS,
        components={
            ElementKind.ELEMENT,
            ElementKind.GROUP,
            ElementKind.SITE_GROUP,
            ElementKind.EDGE_GROUP,
            ElementKind.SYSTEM,
        },
        noise=set(NoiseKind),
        orders={OrderKind.INTEGER},
    )

    renews: ClassVar[bool] = True

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
        """Return the state after one Euler-Maruyama step with given Wiener increments, before the boundary laws.

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
        """
        problem = self.problem_
        following = state + cast('VectorFunction', problem.drift)(time, state, p) * step
        np.add.at(following, problem.noise_rows, cast('VectorFunction', problem.diffusion)(time, state, p) * noise)
        return following

    def _step(self, time: float, state: Vector, p: Vector, lengths: Vector) -> Vector:
        """Return the state after one Euler-Maruyama step and the boundary laws, with one step length per state."""
        noise = self.problem_.draw_increments(self.rng_, 1.0) * np.sqrt(lengths[self.problem_.noise_rows])
        end = time + float(np.max(lengths))
        return self._bounded(end, self._increment(time, state, p, step=lengths, noise=noise, end=end), p)

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate the drift and the noise from time `t` to time `until`.

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

        Raises:
            SolverError:
                The internal step is not positive.
        """
        if not self.dt > 0:
            error_msg = f'dt = {self.dt!r} must be positive.'
            raise SolverError(error_msg)
        drift, diffusion = self.problem_.drift, self.problem_.diffusion
        if until <= t or drift is None or diffusion is None:
            return x
        if self.problem_.renewals:
            return self._advance_renewals(t, x, p, until, dt=self.dt, step=self._step)
        if is_hybrid(self.problem_, t, until):
            return self._advance_hybrid(t, x, p, until)
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.euler_maruyama is not None:
            return accelerated.euler_maruyama(t, x, p, until, self.dt, self.rng_)
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        state = x.copy()
        for index in range(n_steps):
            time = t + index * step
            noise = self.problem_.draw_increments(self.rng_, math.sqrt(step))
            following = self._increment(time, state, p, step=step, noise=noise, end=time + step)
            state = self._bounded(time + step, following, p)
        return state


class MilsteinSolver(SDESolver):
    """Derivative-free Milstein scheme of Platen for SDEs with diagonal noise, of strong order 1.

    It takes a supporting value `Y = X + f h + g sqrt(h)` and corrects the noise with
    `(g(Y) - g(X)) (dW^2 - h) / (2 sqrt(h))`, which needs no derivative of the noise.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.economics.finance import geometric_brownian_motion
        >>> from skcomplex.engine.solvers import MilsteinSolver
        >>> problem = compile_system(geometric_brownian_motion())
        >>> stepper = MilsteinSolver(dt=0.01).prepare(problem, np.random.default_rng(0))
        >>> bool(stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0] > 0)
        True
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        space=set(SpaceKind.registered()),
        laws={LawKind.RATE, LawKind.EQUATION, LawKind.EVENT, LawKind.BOUNDARY, LawKind.TRANSFER, LawKind.NOISE},
        domains=NUMERIC_DOMAINS,
        components={
            ElementKind.ELEMENT,
            ElementKind.GROUP,
            ElementKind.SITE_GROUP,
            ElementKind.EDGE_GROUP,
            ElementKind.SYSTEM,
        },
        noise={NoiseKind.DIAGONAL},
        orders={OrderKind.INTEGER},
    )

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
        """Return the state after one derivative-free Milstein step with given Wiener increments.

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
        """
        problem = self.problem_
        drift = problem.drift(time, state, p)  # type: ignore[misc]
        amplitudes = problem.diffusion(time, state, p)  # type: ignore[misc]
        root = np.sqrt(step)
        support = state + drift * step
        np.add.at(support, problem.noise_rows, amplitudes * root)
        correction = (problem.diffusion(time, support, p) - amplitudes) * (noise**2 - step) / (2 * root)  # type: ignore[misc]
        following = state + drift * step
        np.add.at(following, problem.noise_rows, amplitudes * noise + correction)
        return following


class HeunSolver(SDESolver):
    """Predictor-corrector scheme for SDEs that averages the drift over the step and keeps the Itô noise.

    The predictor `X~ = X + f(X) h + g(X) dW` gives the corrector `X + (f(X) + f(X~)) h / 2 + g(X) dW`, which has strong
    order 1 for additive noise and integrates the Itô form for any noise.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.economics.finance import geometric_brownian_motion
        >>> from skcomplex.engine.solvers import HeunSolver
        >>> problem = compile_system(geometric_brownian_motion())
        >>> stepper = HeunSolver(dt=0.01).prepare(problem, np.random.default_rng(0))
        >>> bool(stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0] > 0)
        True
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        space=set(SpaceKind.registered()),
        laws={LawKind.RATE, LawKind.EQUATION, LawKind.EVENT, LawKind.BOUNDARY, LawKind.TRANSFER, LawKind.NOISE},
        domains=NUMERIC_DOMAINS,
        components={
            ElementKind.ELEMENT,
            ElementKind.GROUP,
            ElementKind.SITE_GROUP,
            ElementKind.EDGE_GROUP,
            ElementKind.SYSTEM,
        },
        noise=set(NoiseKind),
        orders={OrderKind.INTEGER},
    )

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
        """Return the state after one Heun step with given Wiener increments, with the corrector drift at the end.

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
        """
        problem = self.problem_
        drift = problem.drift(time, state, p)  # type: ignore[misc]
        diffusion = np.zeros_like(state)
        np.add.at(diffusion, problem.noise_rows, problem.diffusion(time, state, p) * noise)  # type: ignore[misc]
        predictor = state + drift * step + diffusion
        return state + (drift + problem.drift(end, predictor, p)) * step / 2 + diffusion  # type: ignore[misc]


register_solver('euler-maruyama', EulerMaruyamaSolver, default=True)
register_solver('milstein', MilsteinSolver)
register_solver('heun', HeunSolver)
