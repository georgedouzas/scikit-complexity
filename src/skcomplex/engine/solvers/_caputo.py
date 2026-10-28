"""Base class of steppers of Caputo fractional differential equations on a fixed grid from the start of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod
from typing import ClassVar

import numpy as np

from ...core.errors import SolverError
from ...core.laws import BoundaryLaw, EquationLaw, FluxLaw, RateLaw, TransferLaw
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...core.types import Vector
from ..problems import Form, OrderKind, Problem, SupportSpec
from ._solver import Solver

GRID_TOLERANCE = 1e-9


_CHANGE_TOLERANCE = 1e-12


class CaputoSolver(Solver):
    """Base class of steppers of Caputo fractional differential equations on a fixed grid from the start of a run.

    A state of order `alpha` follows `D^alpha x = f(t, x)`, and a state of order 1 follows `dx/dt = f(t, x)`. The
    stepper keeps the history of the run on the grid `t0 + k dt`. A change of a fractional state between frames starts
    a new history from the changed state, and a change of a state of order 1 carries on.
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={RateLaw, FluxLaw, EquationLaw, BoundaryLaw, TransferLaw},
        orders=set(OrderKind),
    )

    form: ClassVar[Form] = Form.FRACTIONAL
    forms: ClassVar[frozenset[Form]] = frozenset({Form.ODE})

    def _prepare(self, problem: Problem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem of the fractional form, or of the ODE form whose states all have order 1.

            rng:
                The random number generator.

            until:
                The end time of the run.

        Raises:
            SolverError:
                The problem has another form, delays, events, or renewal laws.
        """
        super()._prepare(problem, rng, until)
        if problem.delays or problem.events:
            error_msg = f'{type(self).__name__} does not integrate delays or events.'
            raise SolverError(error_msg)
        self.orders_ = problem.orders if problem.orders.size else np.ones(problem.states.size)
        self._fractional = self.orders_ != 1
        self._unique, self._groups = np.unique(self.orders_, return_inverse=True)
        self._origin = math.nan
        self._step_index = 0
        self._state = np.zeros(0)
        self._initial = np.zeros(0)

    def _index(self, time: float) -> int:
        """Return the index of a time on the grid of the run, after checking that it lies on it."""
        position = (time - self._origin) / self.dt
        index = round(position)
        if abs(position - index) > GRID_TOLERANCE * max(1.0, abs(position)):
            error_msg = (
                f'{type(self).__name__} steps on multiples of dt = {self.dt} from t = {self._origin}, but was asked '
                f'for t = {time}. Make the time between frames a multiple of dt.'
            )
            raise SolverError(error_msg)
        return int(index)

    def _continues(self, t: float, x: Vector) -> bool:
        """Check whether a run goes on from the last step with the same fractional states."""
        if not self._state.size or not math.isclose(t, self._origin + self._step_index * self.dt, abs_tol=1e-12):
            return False
        fractional = self._fractional
        return bool(np.allclose(x[fractional], self._state[fractional], rtol=_CHANGE_TOLERANCE, atol=0.0))

    def _convolve(self, weights: Vector, rows: Vector) -> Vector:
        """Return the sum over rows of every column weighted by the column of its order."""
        if self._unique.size == 1:
            return weights[:, 0] @ rows
        result = np.zeros(rows.shape[1])
        for group in range(self._unique.size):
            columns = self._groups == group
            result[columns] = weights[:, group] @ rows[:, columns]
        return result

    def _per_state(self, values: Vector) -> Vector:
        """Return one value per state from one value per distinct order."""
        return values[self._groups]

    @abstractmethod
    def _restart(self, t: float, x: Vector) -> None:
        """Start the history of a run at a time from a state.

        Args:
            t:
                The time of the start.

            x:
                The state at the start.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _carry(self, x: Vector) -> None:
        """Take the values of the states of order 1 that changed between frames.

        Args:
            x:
                The state given at the start of a frame.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _step(self, p: Vector) -> None:
        """Advance the run by one step of the grid.

        Args:
            p:
                The parameter vector.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate from time `t` to time `until` on the grid of the run.

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
                The internal step is not positive, or `until` is not on the grid of the run.
        """
        if not self.dt > 0:
            error_msg = f'dt = {self.dt!r} must be positive.'
            raise SolverError(error_msg)
        if until <= t or self.problem_.drift is None:
            return x
        if self._continues(t, x):
            self._carry(x)
        else:
            self._origin, self._step_index = t, 0
            self._initial, self._state = x.copy(), x.copy()
            self._restart(t, x)
        stop = self._index(until)
        while self._step_index < stop:
            self._step(p)
            self._step_index += 1
        return self._state.copy()
