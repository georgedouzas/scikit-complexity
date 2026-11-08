"""Euler-Maruyama integrator of stochastic differential equations driven by fractional and ordinary noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.linalg

from ...components.laws import (
    BoundaryLaw,
    CrossingLaw,
    EquationLaw,
    FluxLaw,
    FractionalNoiseLaw,
    NoiseLaw,
    RateLaw,
    TransferLaw,
)
from ...components.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError, ValidationError
from ...specs import SupportSpec
from ...types import Array, FormKind, NoiseKind, OrderKind
from ..problems import ProblemSpec
from ._horizon_error import HorizonError
from ._solver import Solver

_EIGENVALUE_TOLERANCE = 1e-10


GRID_TOLERANCE = 1e-9


_MAX_INCREMENTS = 100_000_000


class FractionalEulerMaruyamaSolver(Solver):
    """Euler-Maruyama integrator of stochastic differential equations driven by fractional and ordinary noise.

    The stepper draws the correlated increments of every fractional noise row for the whole run when it first advances,
    on the grid `t0 + k dt` up to the horizon, and adds the stored increment of each step times the amplitude.
    Ordinary noise rows take independent Wiener increments. A simulation passes the end time of its run as the
    horizon when the stepper has none.

    Args:
        dt:
            The fixed internal step. Frames must fall on multiples of it.

        horizon:
            The last time the run reaches, or `None` for the end time of the simulation's run.

        max_increments:
            The largest number of stored increments, one per step, fractional noise row, and group member.

        random_state:
            The seed or generator of stochastic solvers.

        until:
            The end time of the run the solver prepares for, infinite for a run that goes on until it is stopped.

    Attributes:
        horizon_:
            The horizon of the drawn increments.

        hurst_:
            The Hurst index of every noise row the increments were drawn with.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, GroupElement
        >>> from skcomplex.core.components.laws import FractionalNoiseLaw
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.engine.solvers import FractionalEulerMaruyamaSolver
        >>> from skcomplex.core.specs import StepSpec
        >>> class WalkerElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, s):
        ...         return [FractionalNoiseLaw(s.x, 1, hurst=0.75)]
        >>> system = System(elements=[GroupElement('walkers', element=WalkerElement, size=500)])
        >>> problem = simulate(system, until=0).problem
        >>> stepper = FractionalEulerMaruyamaSolver(dt=0.01, horizon=1.0, random_state=0).prepare(problem)
        >>> state = stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0)).state_
        >>> round(float(np.var(state)), 1)
        1.0
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={RateLaw, FluxLaw, EquationLaw, CrossingLaw, BoundaryLaw, TransferLaw, NoiseLaw, FractionalNoiseLaw},
            noise=frozenset({NoiseKind.DIAGONAL}),
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.SDE

    @classmethod
    def _is_fractional(cls) -> bool:
        """Return whether the solver integrates fractional noise."""
        return True

    def __init__(
        self,
        dt: float = 1e-3,
        horizon: float | None = None,
        max_increments: int = _MAX_INCREMENTS,
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.dt = dt
        self.horizon = horizon
        self.max_increments = max_increments
        self.random_state = random_state
        self.until = until

    def _prepare(self, context: ProblemSpec | None) -> None:
        """Prepare the stepper for a problem.

        Args:
            context:
                The compiled problem.

        Raises:
            SolverError:
                The problem has another form, delays, events, or renewal laws.
        """
        super()._prepare(context)
        problem = self.problem_
        if problem.delays or problem.events:
            error_msg = 'FractionalEulerMaruyamaSolver does not integrate delays or events.'
            raise SolverError(error_msg)
        self.horizon_ = self.horizon if self.horizon is not None else self.until
        self.hurst_: Array | None = None
        self._origin = 0.0
        self._increments: npt.NDArray[np.float64] | None = None
        self._ordinary = np.ones(problem.noise_rows.size, dtype=bool)

    def _hurst(self, p: Array) -> Array:
        """Return the Hurst index of every noise row, 1/2 for ordinary rows."""
        hurst = np.full(self.problem_.noise_rows.size, 0.5)
        for rows in self.problem_.fractional:
            hurst[rows.positions] = rows.hurst(p)
        return hurst

    def _draw(self, t: float, p: Array) -> npt.NDArray[np.float64]:
        """Draw the scaled increments of every fractional noise row from a time up to the horizon."""
        horizon = self.horizon_
        if horizon is None or not math.isfinite(horizon):
            error_msg = (
                f'FractionalEulerMaruyamaSolver draws the noise of a whole run, but the run has the horizon {horizon}. '
                'Give '
                'FractionalEulerMaruyamaSolver(horizon=...) a finite horizon, or run the simulation to a finite end '
                'time.'
            )
            raise HorizonError(error_msg)
        hurst = self._hurst(p)
        mask = np.zeros(hurst.size, dtype=bool)
        for rows in self.problem_.fractional:
            mask[rows.positions] = True
        fractional = np.flatnonzero(mask)
        steps = max(1, math.ceil((horizon - t) / self.dt - GRID_TOLERANCE))
        size = steps * fractional.size
        if size > self.max_increments:
            error_msg = (
                f'FractionalEulerMaruyamaSolver needs {size} increments, {8 * size / 1e6:.1f} MB, for the horizon '
                f'{horizon}, '
                f'above max_increments = {self.max_increments}. Shorten the horizon, take a larger dt, or raise '
                'max_increments.'
            )
            raise SolverError(error_msg)
        increments = np.zeros((steps, hurst.size))
        for value in np.unique(hurst[fractional]):
            columns = fractional[hurst[fractional] == value]
            noise = self._fractional_gaussian_noise(steps, float(value), columns.size, self.rng_)
            increments[:, columns] = noise * self.dt**value
        self._origin, self._increments, self._ordinary, self.hurst_ = t, increments, ~mask, hurst
        return increments

    def _grid_index(self, time: float) -> int:
        """Return the index of a time on the grid of the drawn increments, after checking that it lies on it."""
        position = (time - self._origin) / self.dt
        index = round(position)
        if abs(position - index) > GRID_TOLERANCE * max(1.0, abs(position)):
            error_msg = (
                f'FractionalEulerMaruyamaSolver steps on multiples of dt = {self.dt} from t = {self._origin}, but was '
                'asked '
                f'for t = {time}. Make the time between frames a multiple of dt.'
            )
            raise SolverError(error_msg)
        return int(index)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate the drift and the noise from time `t` to time `until` on the grid of the drawn increments.

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
            HorizonError:
                The run has no finite horizon, or `until` lies beyond it.

            SolverError:
                The internal step is not positive, the increments would exceed `max_increments`, `until` is not on
                the grid, or a Hurst index changed after the increments were drawn.
        """
        if not self.dt > 0:
            error_msg = f'dt = {self.dt!r} must be positive.'
            raise SolverError(error_msg)
        problem = self.problem_
        if until <= t or problem.drift is None or problem.diffusion is None:
            return x
        increments = self._draw(t, p) if self._increments is None else self._increments
        last = self._origin + increments.shape[0] * self.dt
        if until > last + GRID_TOLERANCE * max(1.0, abs(last)):
            error_msg = (
                f'The fractional noise was drawn up to the horizon {self.horizon_}, but the run asked for t = {until}. '
                'Give FractionalEulerMaruyamaSolver a larger horizon.'
            )
            raise HorizonError(error_msg)
        if self.hurst_ is not None and not np.array_equal(self._hurst(p), self.hurst_):
            error_msg = 'A Hurst index changed after the fractional noise was drawn. Start a new run to change it.'
            raise SolverError(error_msg)
        ordinary = self._ordinary
        rows, root = problem.noise_rows, math.sqrt(self.dt)
        state = x.copy()
        for index in range(self._grid_index(t), self._grid_index(until)):
            time = self._origin + index * self.dt
            noise = increments[index].copy()
            noise[ordinary] = self.rng_.standard_normal(int(np.count_nonzero(ordinary))) * root
            following = state + problem.drift(time, state, p) * self.dt
            np.add.at(following, rows, problem.diffusion(time, state, p) * noise)
            state = self._bounded(time + self.dt, following, p)
        return state

    @staticmethod
    def _autocovariance(n: int, hurst: float) -> Array:
        """Return the autocovariance of unit fractional Gaussian noise at lags 0 to `n`."""
        lags = np.arange(n + 1, dtype=float)
        exponent = 2 * hurst
        return 0.5 * (np.abs(lags + 1) ** exponent - 2 * lags**exponent + np.abs(lags - 1) ** exponent)

    @staticmethod
    def _fractional_gaussian_noise(n: int, hurst: float, size: int, rng: np.random.Generator) -> Array:
        """Return paths of unit fractional Gaussian noise, the increments of fractional Brownian motion on a unit grid.

        The paths come from the circulant embedding of Davies and Harte, whose only costly step is an FFT of
        `scipy.fft`. When the embedding has a negative eigenvalue, the paths come from the Cholesky factor of the
        covariance instead.

        Args:
            n:
                The number of increments of every path.

            hurst:
                The Hurst index, between 0 and 1.

            size:
                The number of independent paths.

            rng:
                The random number generator.

        Returns:
            noise:
                One row per increment and one column per path, with unit variance and the covariance
                `(|k + 1|^(2H) - 2 |k|^(2H) + |k - 1|^(2H)) / 2` between increments `k` apart.

        Raises:
            ValidationError:
                The Hurst index is not between 0 and 1, or the number of increments is not positive.

        Examples:
            >>> import numpy as np
            >>> from skcomplex.core.engine.solvers import FractionalEulerMaruyamaSolver
            >>> solver = FractionalEulerMaruyamaSolver
            >>> noise = solver._fractional_gaussian_noise(100, 0.75, 2000, np.random.default_rng(0))
            >>> noise.shape
            (100, 2000)
            >>> round(float(np.mean(noise[1:] * noise[:-1])), 1)
            0.4
        """
        if not 0 < hurst < 1 or n < 1:
            error_msg = f'Fractional Gaussian noise takes a Hurst index in (0, 1) and n >= 1, but got {hurst} and {n}.'
            raise ValidationError(error_msg)
        covariance = FractionalEulerMaruyamaSolver._autocovariance(n, hurst)
        row = np.concatenate([covariance, covariance[-2:0:-1]])
        eigenvalues = scipy.fft.fft(row).real
        if np.min(eigenvalues) < -_EIGENVALUE_TOLERANCE * np.max(eigenvalues):
            factor = scipy.linalg.cholesky(scipy.linalg.toeplitz(covariance[:n]), lower=True)
            return np.asarray(factor @ rng.standard_normal((n, size)), dtype=float)
        scale = np.sqrt(np.clip(eigenvalues, 0.0, None) / row.size)[:, np.newaxis]
        gaussian = rng.standard_normal((row.size, size)) + 1j * rng.standard_normal((row.size, size))
        return np.asarray(scipy.fft.fft(scale * gaussian, axis=0)[:n].real, dtype=float)
