"""Velocity Verlet integrator of particle systems.

The positions of the particles move with their velocities.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...components.laws import BoundaryLaw, EquationLaw, FluxLaw, RateLaw
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from .._utils import components_of
from ..problems import ProblemSpec
from ..problems._steps import step_count
from ._base import BaseSolver
from ._solver import Solver


class VelocityVerletSolver(Solver):
    """Velocity Verlet integrator of particle systems.

    The positions of the particles move with their velocities. Every step kicks the velocities by half a step, moves the
    positions, applies the boundary laws, and kicks the velocities again with the accelerations at the new positions, so
    a step costs one drift evaluation. It conserves energy over long runs of conservative forces. With the Numba backend
    every internal step of a frame runs in compiled code.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.spaces import ContinuousSpace
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.components.elements import PointElement
        >>> from skcomplex.core.engine.solvers import VelocityVerletSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> system = System(elements=[PointElement('ball', x_d1=2.0)], space=ContinuousSpace(n_dim=1))
        >>> problem = simulate(system, until=0).problem
        >>> stepper = VelocityVerletSolver(dt=0.1, random_state=0).prepare(problem)
        >>> stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_.round(10).tolist()
        [2.0, 2.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={RateLaw, FluxLaw, EquationLaw, BoundaryLaw},
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
        self.positions_, self.velocities_ = self._position_pairs(problem)

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
            state = self._bounded(time, state, p)
            rates = drift(time, state, p)
            state[self.velocities_] += step / 2 * rates[self.velocities_]
        return state

    @staticmethod
    def _position_pairs(problem: ProblemSpec) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
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
