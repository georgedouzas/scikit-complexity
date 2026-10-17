"""Implement the default steppers of the four forms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import sympy
from scipy.integrate import solve_ivp

from ..compile import Problem, Vector
from ..core import Form, SolverError
from ._base import Stepper, register_solver


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
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> from skcomplex.solvers import ScipyODE
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


class RK4(Stepper):
    """Classical fourth-order Runge-Kutta integrator of ordinary differential equations with a fixed internal step.

    With the Numba backend every internal step of a frame runs in compiled code.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import ExponentialGrowth, Population
        >>> from skcomplex.solvers import RK4
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


def _position_pairs(problem: Problem) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the flat indices of the position components and of their velocities, in matching order."""
    indices: dict[sympy.Symbol, npt.NDArray[np.int64]] = {}
    for entry in problem.states:
        symbols = problem.symbols[entry.name]
        for column, symbol in enumerate(symbols if isinstance(symbols, sympy.MatrixBase) else [symbols]):
            indices[symbol] = entry.start + np.arange(entry.rows) * entry.columns + column
    positions, velocities = [], []
    for name, equation in problem.equations.items():
        if isinstance(equation, sympy.Symbol) and equation in indices:
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
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import EuclideanSpace, System
        >>> from skcomplex.models.physics import ConstantForce, Particle
        >>> from skcomplex.solvers import VelocityVerlet
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
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.finance import Asset, GeometricBrownianMotion
        >>> from skcomplex.solvers import EulerMaruyama
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


class Gillespie(Stepper):
    """Exact stochastic simulation of jump processes by the Gillespie direct method.

    An event that would fire after `until` is discarded, which is exact because waiting times are memoryless.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.epidemiology import Compartment, Recovery
        >>> from skcomplex.solvers import Gillespie
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
        >>> from skcomplex.compile import compile_system
        >>> from skcomplex.model import System
        >>> from skcomplex.models.economics import Cobweb, Market
        >>> from skcomplex.solvers import Iteration
        >>> problem = compile_system(
        ...     System(elements=[Market('m', price=1)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)]),
        ... )
        >>> stepper = Iteration()
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0).tolist()
        [4.0]
    """

    form: ClassVar[Form] = Form.DISCRETE

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
register_solver('iteration', Iteration, default=True)
