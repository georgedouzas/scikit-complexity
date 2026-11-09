"""Yoshida integrator of fourth order.

It takes three velocity Verlet steps over two blocks of states.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections import deque

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ...components.laws import BoundaryLaw, EquationLaw, RateLaw
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import SupportSpec
from ...types import Array, FormKind, OrderKind
from .._utils import components_of
from ..problems import ProblemSpec
from ..problems._steps import step_count
from ._base import BaseSolver
from ._solver import Solver

OUTER_WEIGHT = 1 / (2 - 2 ** (1 / 3))
WEIGHTS = np.array([OUTER_WEIGHT, 1 - 2 * OUTER_WEIGHT, OUTER_WEIGHT])


class Yoshida4Solver(Solver):
    """Yoshida integrator of fourth order.

    It takes three velocity Verlet steps over two blocks of states. The solver splits the states into two blocks whose
    rates each read only the other block: a state of order 2 into its position and velocity, and states of order 1 by
    the sparsity of the Jacobian of the drift, as the coordinates and the momenta of a separable Hamiltonian split.
    Every step composes three velocity Verlet steps of the lengths `w1 h`, `w0 h`, and `w1 h`, with `w1 = 1 / (2 -
    2^(1/3))` and `w0 = 1 - 2 w1`, which kick the second block by half a step, drift the first, and kick the second
    again. The scheme is symplectic, so the energy of a separable Hamiltonian stays bounded over long runs. With the
    Numba backend every internal step of a frame runs in compiled code.

    Args:
        dt:
            The largest internal time step.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        first_:
            The flat indices of the states of the block that drifts.

        second_:
            The flat indices of the states of the block that kicks.

    Raises:
        SolverError:
            The problem is not an ODE, or its states admit no split into two blocks whose rates read only the other.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.spaces import ContinuousSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.specs import StateSpec
        >>> class BallElement(Element):
        ...     def __init__(
        ...         self, label, x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0, x_d1=0.0
        ...     ):
        ...         super().__init__(label)
        ...         self.x = x
        ...         self.x_d1 = x_d1
        >>> from skcomplex.core.engine.solvers import Yoshida4Solver
        >>> from skcomplex.core.specs import StepSpec
        >>> system = CompositeSystem(elements=[BallElement('ball', x_d1=2.0)], space=ContinuousSpace(n_dim=1))
        >>> problem = simulate(system, until=0).problem
        >>> stepper = Yoshida4Solver(dt=0.1).prepare(problem)
        >>> stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_.round(10).tolist()
        [2.0, 2.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={RateLaw, EquationLaw, BoundaryLaw},
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
        """Prepare the stepper and split the states of the problem into two blocks.

        One block drifts and the other kicks.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not an ODE, or its states admit no split.
        """
        super()._prepare(context)
        problem = self.problem_
        BaseSolver._reject_hybrid(problem, type(self).__name__)
        self.first_, self.second_ = self._split(problem)

    def _get_reads(self, problem: ProblemSpec) -> list[set[int]]:
        """Return the state components the rate of every component reads.

        They come from the Jacobian and the own equations.
        """
        size = problem.initial_state.size
        jacobian = problem.jacobian
        pattern = None if jacobian is None else jacobian.pattern()
        if pattern is None:
            error_msg = (
                'Yoshida4Solver splits the states by the sparsity of the Jacobian, which the problem does not state. '
                'Use RK4Solver or ScipyODESolver for this system.'
            )
            raise SolverError(error_msg)
        rows, columns = pattern.nonzero()
        reads: list[set[int]] = [set() for _ in range(size)]
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if row != column:
                reads[row].add(column)
        indices = {}
        for entry in problem.states.entries:
            for column, item in enumerate(components_of(problem.symbols[entry.name])):
                indices[item] = entry.start + np.arange(entry.rows) * entry.columns + column
        for name, equation in problem.equations.items():
            item = problem.equation_symbols[name]
            if item in indices and SR(equation).has(item):
                for index in indices[item].tolist():
                    reads[index].add(index)
        return reads

    def _get_names(self, problem: ProblemSpec) -> list[str]:
        """Return the name of every state component by its flat index."""
        names = [''] * problem.initial_state.size
        for entry in problem.states.entries:
            names[entry.start : entry.start + entry.size] = entry.component_names
        return names

    def _color(self, reads: list[set[int]]) -> list[int]:
        """Return the block of every component, alternating along the reads.

        Unread components go to the first block.
        """
        neighbors: list[set[int]] = [set(items) for items in reads]
        for index, items in enumerate(reads):
            for item in items:
                neighbors[item].add(index)
        colors = [-1] * len(reads)
        for start in range(len(reads)):
            if colors[start] >= 0:
                continue
            colors[start] = 0
            queue = deque([start])
            while queue:
                index = queue.popleft()
                for item in sorted(neighbors[index]):
                    if colors[item] < 0:
                        colors[item] = 1 - colors[index]
                        queue.append(item)
        return colors

    def _split(self, problem: ProblemSpec) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the flat indices of the block that drifts and of the block that kicks.

        An error is raised when the states cannot be split.
        """
        reads = self._get_reads(problem)
        colors = self._color(reads)
        conflicts = [{item for item in items if colors[item] == colors[index]} for index, items in enumerate(reads)]
        worst = max(range(len(reads)), key=lambda index: len(conflicts[index]), default=None)
        if worst is not None and conflicts[worst]:
            names = self._get_names(problem)
            read = ', '.join(names[item] for item in sorted(conflicts[worst]))
            error_msg = (
                f'Yoshida4Solver splits the states into two blocks whose rates read only the other block, but the rate '
                f'of {names[worst]} reads {read} of its own block. Use a solver that needs no split, such as RK4Solver.'
            )
            raise SolverError(error_msg)
        first = [index for index, color in enumerate(colors) if color == 0]
        second = [index for index, color in enumerate(colors) if color == 1]
        return np.array(first, dtype=np.int64), np.array(second, dtype=np.int64)

    def _stage(self, time: float, state: Array, p: Array, rates: Array, substep: float) -> tuple[Array, Array]:
        """Return the state and rates after one velocity Verlet step of a length.

        The step starts from the rates at its start.
        """
        drift = self.problem_.drift
        assert drift is not None
        state[self.second_] += substep / 2 * rates[self.second_]
        rates = drift(time, state, p)
        state[self.first_] += substep * rates[self.first_]
        state = self._bounded(time + substep, state, p)
        rates = drift(time + substep, state, p)
        state[self.second_] += substep / 2 * rates[self.second_]
        return state, rates

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate the two blocks from time `t` to time `until` in equal steps.

        The steps are no longer than `dt`.

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
        if accelerated is not None and accelerated.yoshida4 is not None:
            return accelerated.yoshida4(t, x, p, until, self.dt, self.first_, self.second_, WEIGHTS)
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        state = x.copy()
        rates = drift(t, state, p)
        for index in range(n_steps):
            time = t + index * step
            for weight in WEIGHTS:
                state, rates = self._stage(time, state, p, rates, weight * step)
                time += weight * step
        return state
