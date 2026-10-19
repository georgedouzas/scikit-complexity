"""Implement the default steppers of the four forms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, OdeSolver, Radau, solve_ivp

from ..base import Expression, Form, SolverError, components_of
from ..compile import Problem, Vector
from ._base import Stepper, register_solver
from ._hybrid import History, hermite, with_delays

METHODS = {'RK45': RK45, 'RK23': RK23, 'DOP853': DOP853, 'Radau': Radau, 'BDF': BDF, 'LSODA': LSODA}
RESTART = 1e-12


def _is_hybrid(problem: Problem, t: float, until: float) -> bool:
    """Check whether a problem needs the hybrid path between two times: delays, events, or breakpoints up to the end."""
    return bool(problem.delays or problem.events or problem.breakpoints.between(t, math.nextafter(until, math.inf)))


def _smallest_delay(problem: Problem) -> float:
    """Return the smallest delay of a problem, or infinity without delays."""
    return min((delay.minimum for delay in problem.delays), default=math.inf)


def _bounds(problem: Problem, t: float, until: float) -> list[float]:
    """Return the breakpoints between two times followed by the end time."""
    return [*problem.breakpoints.between(t, until), until]


def _left_limit(bound: float) -> float:
    """Return the last time before a bound, where inputs that jump at the bound still take their earlier value."""
    return float(np.nextafter(bound, -math.inf))


def _unsupported(stepper: Stepper, problem: Problem) -> None:
    """Raise an error when a stepper without a hybrid path receives delays or events."""
    if problem.delays or problem.events:
        error_msg = (
            f'{type(stepper).__name__} does not integrate delays or events. Use ScipyODE or RK4 for ODEs and '
            'EulerMaruyama, Milstein, or Heun for SDEs.'
        )
        raise SolverError(error_msg)


class ScipyODE(Stepper):
    """Integrator of ordinary differential equations based on `scipy.integrate.solve_ivp`.

    Args:
        method:
            The integration method of `solve_ivp`, such as `RK45`, `DOP853`, or `LSODA` for stiff systems.

        rtol:
            The relative tolerance.

        atol:
            The absolute tolerance.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> from skcomplex.core.solvers import ScipyODE
        >>> problem = compile_system(
        ...     System(elements=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1)]),
        ... )
        >>> stepper = ScipyODE(rtol=1e-8)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> x = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> round(float(x[0]), 4)
        2.7183
    """

    form: ClassVar[Form] = Form.ODE

    def __init__(self, method: str = 'RK45', rtol: float = 1e-6, atol: float = 1e-9) -> None:
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate the drift from time `t` to time `until`.

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
                The integrator fails.
        """
        drift = self.problem_.drift
        if until <= t or x.size == 0 or drift is None:
            return x
        if _is_hybrid(self.problem_, t, until):
            return self._advance_hybrid(t, x, p, until)
        solution = solve_ivp(
            lambda time, state: drift(time, state, p),
            (t, until),
            x,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not solution.success:
            error_msg = f'The {self.method} integrator failed between t = {t} and t = {until}: {solution.message}'
            raise SolverError(error_msg)
        return np.asarray(solution.y[:, -1], dtype=float)

    def _advance_hybrid(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate step by step, restarting at breakpoints and events, and recording dense output for delays."""
        problem = self.problem_
        drift = problem.drift
        if problem.delays and self.history_ is None:
            self.history_ = History(problem, t, x)
        history = self.history_

        def field(time: float, state: Vector, ceiling: float) -> Vector:
            time = min(time, ceiling)
            return drift(time, state, with_delays(problem, history, time, state, p))  # type: ignore[misc]

        state, time = x.copy(), t
        for bound in _bounds(problem, t, until):
            ceiling = _left_limit(bound)
            while time < bound - RESTART * max(1.0, abs(bound)):
                solver = METHODS[self.method](
                    lambda s, y, ceiling=ceiling: field(s, y, ceiling),
                    time,
                    state,
                    bound,
                    rtol=self.rtol,
                    atol=self.atol,
                    max_step=_smallest_delay(problem),
                )
                state, time = self._step_to_event(solver, history, p)
                if self.locator_.stopped_at is not None:
                    return state
            time = bound
        return state

    def _step_to_event(self, solver: OdeSolver, history: History | None, p: Vector) -> tuple[Vector, float]:
        """Step an integrator to its end or to the first event, and return the state and the time where it stopped."""
        while solver.status == 'running':
            start, before = solver.t, solver.y.copy()
            message = solver.step()
            if solver.status == 'failed':
                error_msg = f'The {self.method} integrator failed at t = {start}: {message}'
                raise SolverError(error_msg)
            dense = solver.dense_output()
            if history is not None:
                history.add(start, solver.t, dense)
            if not self.problem_.events:
                continue
            hit = self.locator_.crossing(start, solver.t, before, solver.y, p=p, interpolant=dense)
            if hit is not None:
                time, index = hit
                if history is not None:
                    history.truncate(time)
                return self.locator_.apply(time, index, np.asarray(dense(time), dtype=float), p), time
        return solver.y.copy(), solver.t


class RK4(Stepper):
    """Classical fourth-order Runge-Kutta integrator of ordinary differential equations with a fixed internal step.

    With the Numba backend every internal step of a frame runs in compiled code.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> from skcomplex.core.solvers import RK4
        >>> problem = compile_system(
        ...     System(elements=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1)]),
        ... )
        >>> stepper = RK4(dt=0.001)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> round(float(stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0]), 10)
        2.7182818285
    """

    form: ClassVar[Form] = Form.ODE

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate the drift from time `t` to time `until` in equal steps no longer than `dt`.

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
        drift = self.problem_.drift
        if until <= t or drift is None:
            return x
        if _is_hybrid(self.problem_, t, until):
            return self._advance_hybrid(t, x, p, until)
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.rk4 is not None:
            return accelerated.rk4(t, x, p, until, self.dt)
        n_steps = max(1, math.ceil((until - t) / self.dt - 1e-12))
        step = (until - t) / n_steps
        state = x.copy()
        for index in range(n_steps):
            time = t + index * step
            k1 = drift(time, state, p)
            k2 = drift(time + step / 2, state + step / 2 * k1, p)
            k3 = drift(time + step / 2, state + step / 2 * k2, p)
            k4 = drift(time + step, state + step * k3, p)
            state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state

    def _advance_hybrid(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate with RK4 steps through delays, events, and breakpoints."""
        return _advance_rk4_hybrid(self, t, x, p, until, dt=self.dt)


def _rk4_step(field: Callable[[float, Vector], Vector], time: float, state: Vector, step: float) -> Vector:
    """Return the state after one classical Runge-Kutta step."""
    k1 = field(time, state)
    k2 = field(time + step / 2, state + step / 2 * k1)
    k3 = field(time + step / 2, state + step / 2 * k2)
    k4 = field(time + step, state + step * k3)
    return state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _advance_rk4_hybrid(stepper: Stepper, t: float, x: Vector, p: Vector, until: float, *, dt: float) -> Vector:
    """Integrate with fixed RK4 steps, interpolating every step for delays and locating events in it."""
    problem = stepper.problem_
    drift = problem.drift
    if dt > _smallest_delay(problem):
        error_msg = f'dt = {dt} exceeds the smallest delay {_smallest_delay(problem)}. Take a smaller step.'
        raise SolverError(error_msg)
    if problem.delays and stepper.history_ is None:
        stepper.history_ = History(problem, t, x)
    history = stepper.history_

    state, time = x.copy(), t
    for bound in _bounds(problem, t, until):
        ceiling = _left_limit(bound)

        def field(time: float, state: Vector, ceiling: float = ceiling) -> Vector:
            time = min(time, ceiling)
            return drift(time, state, with_delays(problem, history, time, state, p))  # type: ignore[misc]

        while time < bound - RESTART * max(1.0, abs(bound)):
            step = min(dt, bound - time)
            following = _rk4_step(field, time, state, step)
            interpolant = hermite(
                time,
                state,
                field(time, state),
                t1=time + step,
                x1=following,
                f1=field(time + step, following),
            )
            if history is not None:
                history.add(time, time + step, interpolant)
            hit = (
                stepper.locator_.crossing(time, time + step, state, following, p=p, interpolant=interpolant)
                if problem.events
                else None
            )
            if hit is not None:
                time, index = hit
                if history is not None:
                    history.truncate(time)
                state = stepper.locator_.apply(time, index, interpolant(time), p)
                if stepper.locator_.stopped_at is not None:
                    return state
                continue
            state, time = following, time + step
        time = bound
    return state


def _advance_sde_hybrid(
    stepper: Stepper,
    t: float,
    x: Vector,
    p: Vector,
    until: float,
    *,
    dt: float,
    increment: Callable[..., Vector],
) -> Vector:
    """Integrate an SDE with fixed steps of a scheme, reading delays and locating events by linear interpolation."""
    problem = stepper.problem_
    if dt > _smallest_delay(problem):
        error_msg = f'dt = {dt} exceeds the smallest delay {_smallest_delay(problem)}. Take a smaller step.'
        raise SolverError(error_msg)
    if problem.delays and stepper.history_ is None:
        stepper.history_ = History(problem, t, x)
    history = stepper.history_
    state, time = x.copy(), t
    for bound in _bounds(problem, t, until):
        ceiling = _left_limit(bound)
        while time < bound - RESTART * max(1.0, abs(bound)):
            step = min(dt, bound - time)
            parameters = with_delays(problem, history, time, state, p)
            noise = stepper.rng_.normal(0.0, math.sqrt(step), size=problem.noise_rows.size)
            following = increment(time, state, parameters, step=step, noise=noise, end=min(time + step, ceiling))
            start, end = state.copy(), following.copy()

            def interpolant(
                s: float,
                start: Vector = start,
                end: Vector = end,
                time: float = time,
                step: float = step,
            ) -> Vector:
                return start + (s - time) / step * (end - start)

            if history is not None:
                history.add(time, time + step, interpolant)
            hit = (
                stepper.locator_.crossing(time, time + step, state, following, p=p, interpolant=interpolant)
                if problem.events
                else None
            )
            if hit is not None:
                time, index = hit
                if history is not None:
                    history.truncate(time)
                state = stepper.locator_.apply(time, index, interpolant(time), p)
                if stepper.locator_.stopped_at is not None:
                    return state
                continue
            state, time = following, time + step
        time = bound
    return state


def _position_pairs(problem: Problem) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the flat indices of the position components and of their velocities, in matching order."""
    indices: dict[Expression, npt.NDArray[np.int64]] = {}
    for entry in problem.states:
        for column, item in enumerate(components_of(problem.symbols[entry.name])):
            indices[item] = entry.start + np.arange(entry.rows) * entry.columns + column
    positions, velocities = [], []
    for name, equation in problem.equations.items():
        if equation.is_symbol() and equation in indices:
            positions.append(indices[problem.equation_symbols[name]])
            velocities.append(indices[equation])
    position_indices = np.concatenate([np.zeros(0, dtype=np.int64), *positions]).astype(np.int64)
    velocity_indices = np.concatenate([np.zeros(0, dtype=np.int64), *velocities]).astype(np.int64)
    covered = set(position_indices.tolist()) | set(velocity_indices.tolist())
    missing = [
        entry.name for entry in problem.states if not set(range(entry.start, entry.start + entry.size)) <= covered
    ]
    if missing:
        error_msg = (
            f'VelocityVerlet integrates positions and velocities, with dx/dt = v, but {", ".join(missing)} are '
            'neither. Use RK4 or ScipyODE for this system.'
        )
        raise SolverError(error_msg)
    return position_indices, velocity_indices


class VelocityVerlet(Stepper):
    """Velocity Verlet integrator of particle systems, whose positions move with their velocities.

    Every step kicks the velocities by half a step, moves the positions, and kicks the velocities again with the
    accelerations at the new positions, so a step costs one drift evaluation. It conserves energy over long runs of
    conservative forces. With the Numba backend every internal step of a frame runs in compiled code.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import EuclideanSpace, System
        >>> from skcomplex.models.physics import ConstantForce, Particle
        >>> from skcomplex.core.solvers import VelocityVerlet
        >>> problem = compile_system(
        ...     System(
        ...         elements=[Particle('ball')],
        ...         interactions=[ConstantForce('push', target='ball', F=2.0)],
        ...         space=EuclideanSpace(1),
        ...     ),
        ... )
        >>> stepper = VelocityVerlet(dt=0.1)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0).round(10).tolist()
        [1.0, 2.0]
    """

    form: ClassVar[Form] = Form.ODE

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper and find the position and velocity components of the problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator, unused by this deterministic stepper.

        Raises:
            SolverError:
                The problem is not an ODE, or has states that are neither positions nor velocities.
        """
        super().initialize(problem, rng)
        _unsupported(self, problem)
        self.positions_, self.velocities_ = _position_pairs(problem)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate positions and velocities from time `t` to time `until` in equal steps no longer than `dt`.

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
        drift = self.problem_.drift
        if until <= t or drift is None:
            return x
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.velocity_verlet is not None:
            return accelerated.velocity_verlet(t, x, p, until, self.dt, self.positions_, self.velocities_)
        n_steps = max(1, math.ceil((until - t) / self.dt - 1e-12))
        step = (until - t) / n_steps
        state = x.copy()
        rates = drift(t, state, p)
        for index in range(n_steps):
            state[self.velocities_] += step / 2 * rates[self.velocities_]
            state[self.positions_] += step * state[self.velocities_]
            rates = drift(t + (index + 1) * step, state, p)
            state[self.velocities_] += step / 2 * rates[self.velocities_]
        return state


class EulerMaruyama(Stepper):
    """Euler-Maruyama integrator of stochastic differential equations with a fixed internal step.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.finance import Asset, GeometricBrownianMotion
        >>> from skcomplex.core.solvers import EulerMaruyama
        >>> problem = compile_system(
        ...     System(
        ...         elements=[Asset('stock', price=100)],
        ...         interactions=[GeometricBrownianMotion('gbm', target='stock', mu=0.05, sigma=0.2)],
        ...     ),
        ... )
        >>> stepper = EulerMaruyama(dt=1e-3)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> bool(stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0] > 0)
        True
    """

    form: ClassVar[Form] = Form.SDE

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
        if _is_hybrid(self.problem_, t, until):
            return _advance_sde_hybrid(self, t, x, p, until, dt=self.dt, increment=self._increment)
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.euler_maruyama is not None:
            return accelerated.euler_maruyama(t, x, p, until, self.dt, self.rng_)
        n_steps = math.ceil((until - t) / self.dt)
        step = (until - t) / n_steps
        rows = self.problem_.noise_rows
        state = x.copy()
        for index in range(n_steps):
            time = t + index * step
            amplitudes = diffusion(time, state, p)
            increments = self.rng_.normal(0.0, math.sqrt(step), size=amplitudes.size)
            state = state + drift(time, state, p) * step
            np.add.at(state, rows, amplitudes * increments)
        return state

    def _increment(
        self,
        time: float,
        state: Vector,
        p: Vector,
        *,
        step: float,
        noise: Vector,
        end: float,
    ) -> Vector:
        """Return the state after one Euler-Maruyama step with given Wiener increments."""
        problem = self.problem_
        following = state + problem.drift(time, state, p) * step  # type: ignore[misc]
        np.add.at(following, problem.noise_rows, problem.diffusion(time, state, p) * noise)  # type: ignore[misc]
        return following


class Milstein(Stepper):
    """Derivative-free Milstein scheme of Platen for SDEs with diagonal noise, of strong order 1.

    It takes a supporting value `Y = X + f h + g sqrt(h)` and corrects the noise with
    `(g(Y) - g(X)) (dW^2 - h) / (2 sqrt(h))`, which needs no derivative of the noise.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.models.finance import geometric_brownian_motion
        >>> from skcomplex.core.solvers import Milstein
        >>> problem = compile_system(geometric_brownian_motion())
        >>> stepper = Milstein(dt=0.01)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> bool(stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0] > 0)
        True
    """

    form: ClassVar[Form] = Form.SDE

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def _increment(
        self,
        time: float,
        state: Vector,
        p: Vector,
        *,
        step: float,
        noise: Vector,
        end: float,
    ) -> Vector:
        """Return the state after one derivative-free Milstein step."""
        problem = self.problem_
        drift = problem.drift(time, state, p)  # type: ignore[misc]
        amplitudes = problem.diffusion(time, state, p)  # type: ignore[misc]
        root = math.sqrt(step)
        support = state + drift * step
        np.add.at(support, problem.noise_rows, amplitudes * root)
        correction = (problem.diffusion(time, support, p) - amplitudes) * (noise**2 - step) / (2 * root)  # type: ignore[misc]
        following = state + drift * step
        np.add.at(following, problem.noise_rows, amplitudes * noise + correction)
        return following

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
        return _advance_sde_hybrid(self, t, x, p, until, dt=self.dt, increment=self._increment)


class Heun(Stepper):
    """Predictor-corrector scheme for SDEs that averages the drift over the step and keeps the Itô noise.

    The predictor `X~ = X + f(X) h + g(X) dW` gives the corrector `X + (f(X) + f(X~)) h / 2 + g(X) dW`, which has strong
    order 1 for additive noise and integrates the Itô form for any noise.

    Args:
        dt:
            The fixed internal step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.models.finance import geometric_brownian_motion
        >>> from skcomplex.core.solvers import Heun
        >>> problem = compile_system(geometric_brownian_motion())
        >>> stepper = Heun(dt=0.01)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> bool(stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0] > 0)
        True
    """

    form: ClassVar[Form] = Form.SDE

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def _increment(self, time: float, state: Vector, p: Vector, *, step: float, noise: Vector, end: float) -> Vector:
        """Return the state after one Heun step, with the corrector drift at the end of the step."""
        problem = self.problem_
        drift = problem.drift(time, state, p)  # type: ignore[misc]
        diffusion = np.zeros_like(state)
        np.add.at(diffusion, problem.noise_rows, problem.diffusion(time, state, p) * noise)  # type: ignore[misc]
        predictor = state + drift * step + diffusion
        return state + (drift + problem.drift(end, predictor, p)) * step / 2 + diffusion  # type: ignore[misc]

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
        return _advance_sde_hybrid(self, t, x, p, until, dt=self.dt, increment=self._increment)


class Gillespie(Stepper):
    """Exact stochastic simulation of jump processes by the Gillespie direct method.

    An event that would fire after `until` is discarded, which is exact because waiting times are memoryless.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.epidemiology import Compartment, Recovery
        >>> from skcomplex.core.solvers import Gillespie
        >>> problem = compile_system(
        ...     System(
        ...         elements=[Compartment('I', n=10), Compartment('R', n=0)],
        ...         interactions=[Recovery('rec', infected='I', recovered='R', gamma=1.0, stochastic=True)],
        ...     ),
        ... )
        >>> stepper = Gillespie()
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 100.0).tolist()
        [0.0, 10.0]
    """

    form: ClassVar[Form] = Form.JUMP

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator.

        Raises:
            SolverError:
                The problem has another form, delays, or events.
        """
        super().initialize(problem, rng)
        _unsupported(self, problem)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Fire random events from time `t` until the next event falls after `until`.

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
                A propensity is negative or not finite.
        """
        propensities, jumps = self.problem_.propensities, self.problem_.jumps
        if propensities is None:
            return x
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.gillespie is not None:
            try:
                return accelerated.gillespie(t, x, p, until, self.rng_)
            except ValueError as error:
                error_msg = f'Propensities at t = {t} must be finite and non-negative: {error}'
                raise SolverError(error_msg) from error
        offsets = np.array([channels.offset for channels in jumps])
        state = x.copy()
        while True:
            rates = propensities(t, state, p)
            if np.any(rates < 0) or not np.all(np.isfinite(rates)):
                error_msg = f'Propensities {rates.tolist()} at t = {t} must be finite and non-negative.'
                raise SolverError(error_msg)
            total = float(rates.sum())
            if total == 0:
                return state
            t += self.rng_.exponential(1 / total)
            if t > until:
                return state
            event = min(
                int(np.searchsorted(np.cumsum(rates), self.rng_.uniform(0, total), side='right')),
                len(rates) - 1,
            )
            channels = jumps[int(np.searchsorted(offsets, event, side='right')) - 1]
            np.add.at(state, channels.targets[:, event - channels.offset], channels.increments)


class Iteration(Stepper):
    """Iteration of a discrete map, one application per unit of time.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.economics import Cobweb, Market
        >>> from skcomplex.core.solvers import Iteration
        >>> problem = compile_system(
        ...     System(elements=[Market('m', price=1)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)]),
        ... )
        >>> stepper = Iteration()
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0).tolist()
        [4.0]
    """

    form: ClassVar[Form] = Form.DISCRETE

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator.

        Raises:
            SolverError:
                The problem has another form, delays, or events.
        """
        super().initialize(problem, rng)
        _unsupported(self, problem)

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Apply the map once per integer time step from `t` up to `until`.

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
                The state vector after the last whole step.
        """
        update = self.problem_.update
        if update is None:
            return x
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.iterate is not None:
            return accelerated.iterate(t, x, p, until)
        state = x.copy()
        for step in range(round(t), math.floor(until + 1e-9)):
            state = update(float(step), state, p)
        return state


register_solver('scipy', ScipyODE, default=True)
register_solver('rk4', RK4)
register_solver('velocity-verlet', VelocityVerlet)
register_solver('euler-maruyama', EulerMaruyama, default=True)
register_solver('gillespie', Gillespie, default=True)
register_solver('milstein', Milstein)
register_solver('heun', Heun)
register_solver('iteration', Iteration, default=True)
