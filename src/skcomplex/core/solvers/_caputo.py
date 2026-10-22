"""Integrate fractional differential equations of Caputo derivatives with schemes that keep the history of a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from abc import abstractmethod
from typing import ClassVar

import numpy as np
from scipy.special import gamma

from ..base import (
    NUMERIC_DOMAINS,
    ComponentKind,
    Form,
    LawKind,
    OrderKind,
    SolverError,
    SpaceKind,
    Supports,
    TimeKind,
)
from ..compile import Problem, Vector
from ._base import Stepper, register_solver

GRID_TOLERANCE = 1e-9
CHANGE_TOLERANCE = 1e-12
SLOWEST_MODE = 1e-7
FASTEST_MODE = 60.0
MODE_SPACING = 0.35
INITIAL_ROWS = 64


def _grunwald_letnikov_weights(alpha: Vector, count: int) -> Vector:
    """Return the Grünwald-Letnikov weights `w_0` to `w_count`, one column per order."""
    lags = np.arange(1, count + 1, dtype=float)[:, None]
    factors = 1 - (alpha[None, :] + 1) / lags
    return np.vstack([np.ones((1, alpha.size)), np.cumprod(factors, axis=0)])


def _l1_weights(alpha: Vector, count: int) -> Vector:
    """Return the weights `a_0` to `a_count` of the L1 scheme on the displacements from the start, one column per order.

    The weights are the differences `b_j - b_(j-1)` of the L1 weights `b_j = (j + 1)^(1 - alpha) - j^(1 - alpha)` on
    the increments, with `b_0 = 1`, so the sums run over the displacements from the initial state.
    """
    lags = np.arange(count + 1, dtype=float)[:, None]
    exponent = 1 - alpha[None, :]
    increments = (lags + 1) ** exponent - lags**exponent
    increments[0] = 1.0
    return np.vstack([np.ones((1, alpha.size)), np.diff(increments, axis=0)])


def _tail_rates(window: int) -> Vector:
    """Return the decay rates per step of the exponential modes that hold the history older than a window."""
    exponents = np.arange(math.log(SLOWEST_MODE), math.log(FASTEST_MODE / (window + 1)), MODE_SPACING)
    return np.exp(exponents)


def _grunwald_letnikov_tail(alpha: Vector, rates: Vector, window: int) -> Vector:
    """Return the coefficients of the exponential modes whose sum gives the Grünwald-Letnikov weights past a window.

    The weights have the Laplace representation `w_j = -(sin(pi alpha) / pi) int exp(-s j) exp(s alpha)
    (1 - exp(-s))^alpha ds`, discretised with the trapezoidal rule in `log s`.
    """
    s, a = rates[:, None], alpha[None, :]
    density = -np.sin(np.pi * a) / np.pi * np.exp(-s * (window + 1 - a)) * (-np.expm1(-s)) ** a
    return MODE_SPACING * s * density


def _l1_tail(alpha: Vector, rates: Vector, window: int) -> Vector:
    """Return the coefficients of the exponential modes whose sum gives the L1 weights past a window.

    The weights have the Laplace representation `a_j = -((1 - alpha) / Gamma(alpha)) int exp(-s j) s^(alpha - 2)
    (1 - exp(-s))^2 exp(s) ds`, discretised with the trapezoidal rule in `log s`.
    """
    s, a = rates[:, None], alpha[None, :]
    density = -(1 - a) / gamma(a) * s ** (a - 2) * np.expm1(-s) ** 2 * np.exp(-s * window)
    return MODE_SPACING * s * density


def _predictor_weights(alpha: Vector, count: int) -> Vector:
    """Return the weights `(m + 1)^alpha - m^alpha` of the predictor at lags 0 to `count`, one column per order."""
    lags = np.arange(count + 1, dtype=float)[:, None]
    return (lags + 1) ** alpha[None, :] - lags ** alpha[None, :]


def _corrector_weights(alpha: Vector, count: int) -> Vector:
    """Return the weights of the corrector at lags 0 to `count` for every value but the first, one column per order."""
    lags = np.arange(count + 1, dtype=float)[:, None]
    power = alpha[None, :] + 1
    return (lags + 2) ** power + lags**power - 2 * (lags + 1) ** power


class _Rows:
    """Rows of values at consecutive grid steps, kept in one array that grows or keeps only the latest rows."""

    def __init__(self, first: Vector, keep: int | None) -> None:
        self.keep = keep
        self.values = np.zeros((INITIAL_ROWS if keep is None else 2 * keep + 2, first.size))
        self.values[0] = first
        self.start = 0
        self.stop = 1

    def _make_room(self, size: int) -> None:
        """Double the array of a whole history, or move the kept rows of a window to its front."""
        if self.keep is None:
            grown = np.zeros((2 * size, self.values.shape[1]))
            grown[:size] = self.values
            self.values = grown
            return
        kept = self.values[size - self.keep - 1 :].copy()
        self.values[: kept.shape[0]] = kept
        self.start = self.stop - kept.shape[0]

    def append(self, row: Vector) -> None:
        """Add the row of the next step, dropping the rows outside the kept ones."""
        size = self.stop - self.start
        if size == self.values.shape[0]:
            self._make_room(size)
        self.values[self.stop - self.start] = row
        self.stop += 1

    def rows(self, first: int, last: int) -> Vector:
        """Return the rows of the steps `first` to `last - 1`."""
        return self.values[first - self.start : last - self.start]

    def row(self, step: int) -> Vector:
        """Return the row of one step."""
        return self.values[step - self.start]


class CaputoStepper(Stepper):
    """Base class of steppers of Caputo fractional differential equations on a fixed grid from the start of a run.

    A state of order `alpha` follows `D^alpha x = f(t, x)`, and a state of order 1 follows `dx/dt = f(t, x)`. The
    stepper keeps the history of the run on the grid `t0 + k dt`. A change of a fractional state between frames starts
    a new history from the changed state, and a change of a state of order 1 carries on.
    """

    supports: ClassVar[Supports] = Supports(
        time={TimeKind.CONTINUOUS},
        space=set(SpaceKind),
        laws={LawKind.RATE, LawKind.EQUATION, LawKind.BOUNDARY, LawKind.TRANSFER},
        domains=NUMERIC_DOMAINS,
        components={ComponentKind.ELEMENT, ComponentKind.GROUP, ComponentKind.SITE_GROUP, ComponentKind.SYSTEM},
        orders=set(OrderKind),
    )

    form: ClassVar[Form] = Form.FRACTIONAL
    forms: ClassVar[frozenset[Form]] = frozenset({Form.ODE})

    def initialize(self, problem: Problem, rng: np.random.Generator) -> None:
        """Prepare the stepper for a problem.

        Args:
            problem:
                The compiled problem of the fractional form, or of the ODE form whose states all have order 1.

            rng:
                The random number generator.

        Raises:
            SolverError:
                The problem has another form, delays, events, or renewal laws.
        """
        super().initialize(problem, rng)
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
        return bool(np.allclose(x[fractional], self._state[fractional], rtol=CHANGE_TOLERANCE, atol=0.0))

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

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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


class ConvolutionStepper(CaputoStepper):
    """Base class of explicit schemes that weigh the displacements of the states from the start of a run.

    A step sets `g dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^k c_j (x_(k-j) - x_0)` with `c_0 = 1`. With a memory
    window, the displacements older than the window enter through exponential modes fitted to the weights.
    """

    def __init__(self, dt: float = 0.01, memory: float | None = None) -> None:
        self.dt = dt
        self.memory = memory

    @abstractmethod
    def _weights(self, alpha: Vector, count: int) -> Vector:
        """Return the weights `c_0` to `c_count`, one column per order.

        Args:
            alpha:
                The distinct orders.

            count:
                The largest lag.

        Returns:
            weights:
                The weights, one row per lag.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _tail(self, alpha: Vector, rates: Vector, window: int) -> Vector:
        """Return the coefficients of the exponential modes that give the weights past a window, one column per order.

        Args:
            alpha:
                The distinct orders.

            rates:
                The decay rates of the modes per step.

            window:
                The number of steps of the window.

        Returns:
            coefficients:
                The coefficients, one row per mode.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def _gain(self, alpha: Vector) -> Vector:
        """Return the factor `g` of the drift of every order.

        Args:
            alpha:
                The distinct orders.

        Returns:
            gain:
                The factor of every order.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError

    def _restart(self, t: float, x: Vector) -> None:
        """Start the history of displacements, the weights, and the exponential modes of the tail."""
        del t
        if self.memory is not None and not self.memory > 0:
            error_msg = f'memory = {self.memory!r} must be positive or None.'
            raise SolverError(error_msg)
        self._window = None if self.memory is None else max(1, round(self.memory / self.dt))
        self._rows = _Rows(np.zeros(x.size), self._window)
        self._table = self._weights(self._unique, INITIAL_ROWS if self._window is None else self._window)
        self._scale = self._per_state(self._gain(self._unique) * self.dt**self._unique)
        if self._window is not None:
            rates = _tail_rates(self._window)
            self._decay = np.exp(-rates)[:, None]
            self._modes = self._tail(self._unique, rates, self._window)[:, self._groups]
            self._amplitudes = np.zeros((rates.size, x.size))

    def _carry(self, x: Vector) -> None:
        """Replace the last displacement of the states of order 1 by the given state."""
        ordinary = ~self._fractional
        self._state[ordinary] = x[ordinary]
        self._rows.row(self._step_index)[ordinary] = x[ordinary] - self._initial[ordinary]

    def _history_sum(self, step: int) -> Vector:
        """Return the weighted sum of the displacements before a step, with the tail of a memory window."""
        window = self._window
        lags = step - 1 if window is None else min(step - 1, window)
        if lags >= self._table.shape[0]:
            self._table = self._weights(self._unique, 2 * lags)
        total = self._convolve(self._table[lags:0:-1], self._rows.rows(step - lags, step)) if lags else 0.0
        if window is None or step <= window + 1:
            return np.asarray(total + np.zeros(self._state.size))
        self._amplitudes = self._decay * self._amplitudes + self._rows.row(step - window - 1)
        return np.asarray(total + np.sum(self._modes * self._amplitudes, axis=0))

    def _step(self, p: Vector) -> None:
        """Advance the displacements by one explicit step."""
        step = self._step_index + 1
        time = self._origin + self._step_index * self.dt
        drift = self.problem_.drift(time, self._state, p)  # type: ignore[misc]
        displacement = self._scale * drift - self._history_sum(step)
        self._state = self._bounded(time + self.dt, self._initial + displacement, p)
        self._rows.append(self._state - self._initial)


class GrunwaldLetnikov(ConvolutionStepper):
    """Explicit Grünwald-Letnikov scheme of Caputo fractional differential equations, of order 1.

    A step sets `dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^k w_j (x_(k-j) - x_0)` with the weights
    `w_j = (-1)^j binomial(alpha, j)`. States of order 1 take explicit Euler steps.

    Args:
        dt:
            The fixed step. Frames must fall on multiples of it.

        memory:
            The length of time over which the history enters with its exact weights, or `None` for the whole run. The
            history older than the window enters through exponential modes fitted to the weights, so a step costs
            the same however long the run.

    Attributes:
        orders_:
            The order of every state component.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import Rate
        >>> from skcomplex.core.system import System
        >>> from skcomplex.core.solvers import GrunwaldLetnikov
        >>> Relaxing = Element.define('Relaxing', states={'x': state(1.0, order=0.5)}, laws=lambda s: [Rate(s.x, -s.x)])
        >>> problem = compile_system(System(components=[Relaxing('a')]))
        >>> stepper = GrunwaldLetnikov(dt=0.001)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> round(float(stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0]), 2)
        0.43
    """

    def _weights(self, alpha: Vector, count: int) -> Vector:
        """Return the Grünwald-Letnikov weights `w_0` to `w_count`, one column per order."""
        return _grunwald_letnikov_weights(alpha, count)

    def _tail(self, alpha: Vector, rates: Vector, window: int) -> Vector:
        """Return the coefficients of the exponential modes of the Grünwald-Letnikov weights past a window."""
        return _grunwald_letnikov_tail(alpha, rates, window)

    def _gain(self, alpha: Vector) -> Vector:
        """Return the factor 1 of the drift of every order."""
        return np.ones_like(alpha)


class L1Scheme(ConvolutionStepper):
    """Explicit L1 scheme of Caputo fractional differential equations, of order 1.

    The L1 scheme interpolates the state linearly between grid points inside the Caputo integral. A step sets
    `Gamma(2 - alpha) dt^alpha f(t_(k-1), x_(k-1)) = sum_(j=0)^(k-1) b_j (x_(k-j) - x_(k-j-1))` with the weights
    `b_j = (j + 1)^(1 - alpha) - j^(1 - alpha)`. States of order 1 take explicit Euler steps.

    Args:
        dt:
            The fixed step. Frames must fall on multiples of it.

        memory:
            The length of time over which the history enters with its exact weights, or `None` for the whole run. The
            history older than the window enters through exponential modes fitted to the weights, so a step costs
            the same however long the run.

    Attributes:
        orders_:
            The order of every state component.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import Rate
        >>> from skcomplex.core.system import System
        >>> from skcomplex.core.solvers import L1Scheme
        >>> Relaxing = Element.define('Relaxing', states={'x': state(1.0, order=0.5)}, laws=lambda s: [Rate(s.x, -s.x)])
        >>> problem = compile_system(System(components=[Relaxing('a')]))
        >>> stepper = L1Scheme(dt=0.001)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> round(float(stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0]), 2)
        0.43
    """

    def _weights(self, alpha: Vector, count: int) -> Vector:
        """Return the weights `a_0` to `a_count` of the L1 scheme on the displacements, one column per order."""
        return _l1_weights(alpha, count)

    def _tail(self, alpha: Vector, rates: Vector, window: int) -> Vector:
        """Return the coefficients of the exponential modes of the L1 weights past a window."""
        return _l1_tail(alpha, rates, window)

    def _gain(self, alpha: Vector) -> Vector:
        """Return the factor `Gamma(2 - alpha)` of the drift of every order."""
        return np.asarray(gamma(2 - alpha))


class FractionalABM(CaputoStepper):
    """Fractional Adams-Bashforth-Moulton predictor-corrector of Diethelm, Ford, and Freed, of order `1 + alpha`.

    The scheme solves the equivalent Volterra equation `x(t) = x_0 + (1 / Gamma(alpha)) int_0^t (t - s)^(alpha - 1)
    f(s, x(s)) ds`. A rectangle rule over the drifts of the whole run predicts the next state, and a trapezoidal rule
    with the drift at the prediction corrects it. States of order 1 take Heun steps.

    Args:
        dt:
            The fixed step. Frames must fall on multiples of it.

    Attributes:
        orders_:
            The order of every state component.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.base import state
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import Rate
        >>> from skcomplex.core.system import System
        >>> from skcomplex.core.solvers import FractionalABM
        >>> Relaxing = Element.define('Relaxing', states={'x': state(1.0, order=0.5)}, laws=lambda s: [Rate(s.x, -s.x)])
        >>> problem = compile_system(System(components=[Relaxing('a')]))
        >>> stepper = FractionalABM(dt=0.001)
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> round(float(stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0]), 2)
        0.43
    """

    def __init__(self, dt: float = 0.01) -> None:
        self.dt = dt

    def _restart(self, t: float, x: Vector) -> None:
        """Start the history of drifts and the weights of the predictor and the corrector."""
        del t, x
        self._drifts: _Rows | None = None
        self._resize(INITIAL_ROWS)
        alpha = self._unique
        self._predictor_scale = self._per_state(self.dt**alpha / gamma(alpha + 1))
        self._corrector_scale = self._per_state(self.dt**alpha / gamma(alpha + 2))

    def _resize(self, count: int) -> None:
        """Compute the weights of the predictor and the corrector up to a lag."""
        self._predictor = _predictor_weights(self._unique, count)
        self._corrector = _corrector_weights(self._unique, count)

    def _carry(self, x: Vector) -> None:
        """Take the states of order 1 that changed between frames, which the Heun steps start from."""
        ordinary = ~self._fractional
        self._state[ordinary] = x[ordinary]

    def _step(self, p: Vector) -> None:
        """Advance the run by one predictor-corrector step."""
        drift = self.problem_.drift
        step = self._step_index
        time = self._origin + step * self.dt
        current = drift(time, self._state, p)  # type: ignore[misc]
        if self._drifts is None:
            self._drifts = _Rows(current, None)
        else:
            self._drifts.append(current)
        if step + 1 >= self._predictor.shape[0]:
            self._resize(2 * (step + 1))
        drifts = self._drifts.rows(0, step + 1)
        ordinary = ~self._fractional
        predicted = self._initial + self._predictor_scale * self._convolve(self._predictor[step::-1], drifts)
        predicted[ordinary] = self._state[ordinary] + self.dt * current[ordinary]
        alpha = self._unique
        first = self._per_state(step ** (alpha + 1) - (step - alpha) * (step + 1) ** alpha) * drifts[0]
        past = self._convolve(self._corrector[step - 1 :: -1], drifts[1:]) if step else 0.0
        following = drift(time + self.dt, predicted, p)  # type: ignore[misc]
        corrected = self._initial + self._corrector_scale * (following + first + past)
        corrected[ordinary] = self._state[ordinary] + self.dt / 2 * (current[ordinary] + following[ordinary])
        self._state = self._bounded(time + self.dt, corrected, p)


register_solver('grunwald-letnikov', GrunwaldLetnikov)
register_solver('l1', L1Scheme)
register_solver('fractional-abm', FractionalABM, default=True)
