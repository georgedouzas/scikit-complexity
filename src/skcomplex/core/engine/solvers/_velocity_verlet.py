"""Velocity Verlet integrator of particle systems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...components.laws import ConstraintLaw, RateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.spaces import BaseSpace
from ...components.spaces._continuous_time import _ContinuousTime
from ...enums import FormKind, OrderKind
from ...errors import SolverError
from ...types import Array
from .._utils import components_of
from ..backends import SupportSpec
from ..problems import ProblemSpec
from ..problems._steps import step_count
from ._base import BaseSolver
from ._solver import Solver


class VelocityVerletSolver(Solver):
    """A solver of moving particles by velocity Verlet, stable over long runs.

    Every step kicks the velocities by half a step, moves the positions, applies the boundary laws, and kicks the
    velocities again with the accelerations at the new positions, so a step costs one evaluation of the forces. It keeps
    the energy of conservative forces from drifting over long runs, such as a spring that swings from 1 to -1 in half a
    period. With the Numba backend every step of a frame runs in compiled code. Choose it for Newtonian particles over
    long runs, in molecules, planets, and crowds.

    Read more in the user guide, under Solvers.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    See Also:
        Yoshida4Solver: A symplectic method of order 4.
        LammpsSolver: Runs large systems of particles in LAMMPS.

    Examples:
        >>> from skcomplex.core.engine.solvers import VelocityVerletSolver
        >>> from skcomplex.systems.physics.mechanics import HarmonicOscillatorSystem
        >>> system = HarmonicOscillatorSystem(solver=VelocityVerletSolver(dt=0.01)).prepare()
        >>> system.simulate(until=3.1416, dt=3.1416).get_values('mass.x').round(3).tolist()
        [1.0, -1.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_ContinuousTime},
            space={None, BaseSpace},
            laws={RateLaw, ConstraintLaw, _BoundaryLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.ODE

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

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the stepper and find the position and velocity components.

        The components are those of the problem.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not an ODE, or has states that are neither positions nor velocities.
        """
        super()._prepare(context)
        problem = self.problem_
        BaseSolver._reject_hybrid(problem, type(self).__name__)
        self.positions_, self.velocities_ = self._get_position_pairs(problem)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate positions and velocities from time `t` to time `until`.

        Integration takes equal steps no longer than `dt`.

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
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        state = x.copy()
        rates = drift(t, state, p)
        for index in range(n_steps):
            time = t + (index + 1) * step
            state[self.velocities_] += step / 2 * rates[self.velocities_]
            state[self.positions_] += step * state[self.velocities_]
            state = self._get_bounded(time, state, p)
            rates = drift(time, state, p)
            state[self.velocities_] += step / 2 * rates[self.velocities_]
        return state

    @staticmethod
    def _get_position_pairs(problem: ProblemSpec) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the flat indices of the position components and of their velocities.

        They come in matching order.
        """
        indices: dict[Expression, npt.NDArray[np.int64]] = {}
        for entry in problem.states.entries:
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
            entry.name
            for entry in problem.states.entries
            if not set(range(entry.start, entry.start + entry.size)) <= covered
        ]
        if missing:
            error_msg = (
                f'VelocityVerletSolver integrates positions and velocities, with dx/dt = v, but {", ".join(missing)} '
                'are neither. Use RK4Solver or ScipyODESolver for this system.'
            )
            raise SolverError(error_msg)
        return position_indices, velocity_indices
