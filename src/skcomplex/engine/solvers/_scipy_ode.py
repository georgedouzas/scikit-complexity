"""Integrator of ordinary differential equations based on the step solvers of `scipy.integrate`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, OdeSolver, Radau

from ...core.laws import (
    BoundaryLaw,
    CrossingLaw,
    DestructionLaw,
    EncounterLaw,
    EquationLaw,
    FluxLaw,
    ProliferationLaw,
    RateLaw,
    TransferLaw,
)
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...types import Array
from ..problems import BaseValue, Form, OrderKind, Problem, State, SupportSpec
from ._event_locator import EventLocator
from ._history import _RESTART, History
from ._solver import Solver

METHODS = {'RK45': RK45, 'RK23': RK23, 'DOP853': DOP853, 'Radau': Radau, 'BDF': BDF, 'LSODA': LSODA}


_IMPLICIT = frozenset({'Radau', 'BDF', 'LSODA'})


_JACOBIANS = ('auto', 'exact', 'dense')


_DENSITY_LIMIT = 0.5


_BAND_FRACTION = 0.1


_STIFF_SPAN = 1000


_EXPLICIT_SPAN = 100


_DISCONTINUITY_ORDERS = 3


_STEPS_PER_FRAME = 100


_LARGEST_TIME = sys.float_info.max


_LSODA_LARGEST_SIZE = 4096


_STALL_STEP = 1e-9


_RESTING_SPAN = 1e-9


_STALL_STEPS = 1000


_MAX_FRAME_STEPS = 50_000


_BLOW_UP = 1e12


_SMALL_NORM = 1e-5


_TINY_NORM = 1e-15


_FALLBACK_STEP = 1e-6


@dataclass(frozen=True)
class _Segment(BaseValue):
    """Accepted step of an integrator, with its states at both ends and its dense output."""

    start: float
    end: float
    before: Array
    after: Array
    dense: Callable[[float], Array]


class ScipyODESolver(Solver):
    """Integrator of ordinary differential equations based on the step solvers of `scipy.integrate`.

    One integrator runs across the frames of a run, and every frame reads its state from the dense output of the step
    that covers the frame's time, and it restarts at every frame when a condition on time jumps at a time that depends
    on parameters. With delays, steps are at most the shortest delay, or a hundredth of the frame when
    the delay is shorter, and a delay shorter than the step reads the extension of the previous step.

    The implicit methods `BDF`, `Radau`, and `LSODA` receive the structure of the Jacobian that the compiler derives
    from the laws, at every start and restart of the integrator. `BDF` and `Radau` estimate the Jacobian by finite
    differences over groups of independent columns and factor it with sparse LU, which makes them the stiff path for
    finite-difference and finite element fields. An explicit method that spans more than 1000 stable steps of the
    diffusion of a field in one call warns once.

    Args:
        method:
            The integration method: `RK45`, `RK23`, or `DOP853` for non-stiff systems, `BDF` or `Radau` for stiff
            ones, `LSODA`, which starts with Adams methods and switches to BDF methods while the system is stiff, or
            the default `'auto'`. It chooses `LSODA` when its dense work array fits, for at most 4096 states or a
            Jacobian with a narrow band, since `LSODA` stores a dense matrix of the size of the state squared.
            Larger systems get `BDF` with the sparsity pattern of the Jacobian when fields limit the stable step by
            diffusion, and `RK45` otherwise.

        rtol:
            The relative tolerance.

        atol:
            The absolute tolerance.

        jacobian:
            How implicit methods obtain the Jacobian. `'auto'` passes the sparsity pattern to `BDF` and `Radau` and
            the half bandwidths of a pattern narrower than a tenth of the size to `LSODA`, and passes nothing when the
            pattern is denser than one half. `'exact'` passes the exact sparse Jacobian of the laws, which needs laws
            with a symbolic form. `'dense'` passes nothing, so the Jacobian is estimated densely. Explicit methods
            ignore it.

    Attributes:
        method_:
            The method the stepper integrates with, which `'auto'` resolves for the problem.

        jacobian_pattern_:
            The sparsity pattern of the Jacobian that an implicit method uses, or `None`.

        jacobian_options_:
            The keyword arguments of the pattern or the half bandwidths that the integrator receives. The exact
            Jacobian is passed as `jac` at every start of the integrator, with the parameters of the call.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationElement
        >>> from skcomplex.engine.solvers import ScipyODESolver, Step
        >>> problem = compile_system(
        ...     System(
        ...         elements=[PopulationElement('a', n=1)],
        ...         interactions=[ExponentialGrowthInteraction('g', target='a', r=1)],
        ...     ),
        ... )
        >>> stepper = ScipyODESolver(rtol=1e-8, random_state=0).check().prepare(problem)
        >>> x = stepper.run(Step(0.0, problem.initial_state, problem.initial_parameters, 1.0))
        >>> round(float(x[0]), 4)
        2.7183
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={ContinuousTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={
                RateLaw,
                FluxLaw,
                EquationLaw,
                CrossingLaw,
                BoundaryLaw,
                TransferLaw,
                EncounterLaw,
                ProliferationLaw,
                DestructionLaw,
            },
            orders={OrderKind.INTEGER},
        )

    @classmethod
    def _get_form(cls) -> Form:
        """Return the form the solver solves."""
        return Form.ODE

    def __init__(
        self,
        method: str = 'auto',
        rtol: float = 1e-6,
        atol: float = 1e-9,
        jacobian: str = 'auto',
        *,
        random_state: int | np.random.Generator | None = None,
        until: float = math.inf,
    ) -> None:
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.jacobian = jacobian
        self.random_state = random_state
        self.until = until

    def _prepare(self, problem: Problem) -> None:
        """Prepare the stepper for a problem, with no integrator yet.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                The problem is not an ODE, the method or the Jacobian option is unknown, or `LSODA` would store a dense
                matrix larger than its limit.

            ValidationError:
                The exact Jacobian is asked for laws without a symbolic form.
        """
        super()._prepare(problem)
        if self.method != 'auto' and self.method not in METHODS:
            error_msg = f'Unknown method {self.method!r}. Use one of auto, {", ".join(METHODS)}.'
            raise SolverError(error_msg)
        if self.jacobian not in _JACOBIANS:
            error_msg = f'Unknown Jacobian option {self.jacobian!r}. Use one of {", ".join(map(repr, _JACOBIANS))}.'
            raise SolverError(error_msg)
        self.jacobian_pattern_ = None
        self.jacobian_options_: dict[str, Any] = {}
        self._exact = False
        self._warned = False
        self._stalled = 0
        self._frame_steps = 0
        self._choosing = False
        self.method_ = 'LSODA' if self.method == 'auto' else self.method
        if self.method_ in _IMPLICIT and self.jacobian != 'dense' and problem.jacobian is not None:
            self._prepare_jacobian(problem)
        if self.method_ == 'LSODA' and not self._fits_lsoda(problem.initial_state.size):
            if self.method != 'auto':
                error_msg = (
                    f'LSODA would store a dense {problem.initial_state.size} x {problem.initial_state.size} matrix, '
                    f'more than the limit of {_LSODA_LARGEST_SIZE} states without a narrow band. Use '
                    "ScipyODESolver(method='BDF'), which receives the sparsity pattern of the Jacobian, or an explicit "
                    'method.'
                )
                raise SolverError(error_msg)
            self._choosing = problem.stable_step is not None and self.jacobian_pattern_ is not None
            self.method_ = 'RK45'
            self.jacobian_options_ = {}
        self._solver: OdeSolver | None = None
        self._segment: _Segment | None = None
        self._time = math.nan
        self._frame = math.inf
        self._until = math.inf
        self._state = np.zeros(0)
        self._parameters = np.zeros(0)

    def _fits_lsoda(self, size: int) -> bool:
        """Check whether the dense or banded matrix of LSODA fits for a state of a size."""
        return size <= _LSODA_LARGEST_SIZE or 'lband' in self.jacobian_options_

    def _prepare_jacobian(self, problem: Problem) -> None:
        """Derive the Jacobian structure an implicit method receives, or check that the exact Jacobian exists."""
        jacobian = problem.jacobian
        assert jacobian is not None
        if self.jacobian == 'exact':
            jacobian.prepare(problem)
            self._exact = True
            self.jacobian_pattern_ = jacobian.pattern()
            return
        pattern = jacobian.pattern()
        size = problem.initial_state.size
        if pattern is None or pattern.nnz > _DENSITY_LIMIT * size * size:
            return
        self.jacobian_pattern_ = pattern
        if self.method_ != 'LSODA':
            self.jacobian_options_ = {'jac_sparsity': pattern}
            return
        rows, columns = pattern.nonzero()
        lower = int(np.max(rows - columns, initial=0))
        upper = int(np.max(columns - rows, initial=0))
        if max(lower, upper) <= _BAND_FRACTION * size:
            self.jacobian_options_ = {'lband': lower, 'uband': upper}

    def _jacobian_options(self, history: History | None, p: Array) -> dict[str, Any]:
        """Return the keyword arguments that pass the Jacobian structure to the integrator."""
        if not self._exact:
            return self.jacobian_options_
        problem = self.problem_
        jacobian = problem.jacobian
        assert jacobian is not None
        dense = self.method_ == 'LSODA'

        def exact(time: float, state: Array) -> Any:  # noqa: ANN401
            matrix = jacobian.run(State(time, state, History._with_delays(problem, history, time, state, p)))
            return matrix.toarray() if dense else matrix

        return {'jac': exact}

    def _choose_for_span(self, span: float) -> None:
        """Choose RK45 or BDF for a large diffusive system by how many stable steps the first call spans."""
        if not self._choosing:
            return
        self._choosing = False
        limit = self.problem_.stable_step
        if limit is not None and span > _EXPLICIT_SPAN * limit:
            self.method_ = 'BDF'
            self._prepare_jacobian(self.problem_)

    def _warn_stiff(self, t: float, until: float) -> None:
        """Warn once when an explicit method spans many stable steps of the diffusion of a field in one call."""
        limit = self.problem_.stable_step
        if self._warned or self.method_ in _IMPLICIT or limit is None or until - t <= _STIFF_SPAN * limit:
            return
        self._warned = True
        warning_msg = (
            f'{self.method_} spans {until - t:.6g} in one call, more than {_STIFF_SPAN} times the stability limit '
            f'{limit:.6g} of the diffusion of the fields, so its steps are limited by stability. Use '
            "ScipyODESolver(method='BDF') or ScipyODESolver(method='Radau'), which receive the sparsity pattern of the "
            "Jacobian."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=3)

    def _continues(self, t: float, x: Array, p: Array) -> bool:
        """Check whether a call continues the state, time, and parameters the integrator returned last."""
        return (
            self._solver is not None
            and abs(t - self._time) <= _RESTART * max(1.0, abs(t))
            and np.array_equal(x, self._state)
            and np.array_equal(p, self._parameters)
        )

    def _next_discontinuity(self, time: float) -> float:
        """Return the first time after a time where a delay carries the jump in slope at the start of the run."""
        history = self.history_
        if history is None:
            return math.inf
        times = [
            history.start + order * delay.minimum
            for delay in self.problem_.delays
            for order in range(1, _DISCONTINUITY_ORDERS + 1)
            if history.start + order * delay.minimum > time
        ]
        return min(times, default=math.inf)

    def _restart(self, t: float, x: Array, p: Array) -> None:
        """Start a new integrator at a time and state, bounded by the next breakpoint."""
        problem = self.problem_
        drift = problem.drift
        history = self.history_
        if history is not None:
            history.truncate(t)
        after = t + _RESTART * max(1.0, abs(t))
        later = problem.breakpoints.next_after(after)
        if problem.breakpoints.unresolved:
            later = self._until if later is None else min(later, self._until)
        bound = min(later if later is not None else math.inf, self._next_discontinuity(after))
        ceiling = EventLocator._find_left_limit(bound) if math.isfinite(bound) and later == bound else math.inf

        def field(time: float, state: Array) -> Array:
            time = min(time, ceiling)
            return drift(time, state, History._with_delays(problem, history, time, state, p))  # type: ignore[misc]

        delay = EventLocator._find_smallest_delay(problem)
        end = min(bound, _LARGEST_TIME)
        max_step = max(delay, self._frame / _STEPS_PER_FRAME)
        first_step = min(delay / 2, bound - t) if problem.delays else None
        if first_step is None and self.method_ == 'LSODA':
            first_step = self._lsoda_first_step(field, t, x, end=end, max_step=max_step, rtol=self.rtol, atol=self.atol)
        self._solver = METHODS[self.method_](
            field,
            t,
            x.copy(),
            end,
            rtol=self.rtol,
            atol=self.atol,
            max_step=max_step,
            first_step=first_step,
            **self._jacobian_options(history, p),
        )
        self._segment = None
        self._time, self._state, self._parameters = t, x.copy(), p.copy()

    def _next_segment(self, p: Array) -> _Segment:
        """Return the step that covers the current time, taking a new step or restarting at a breakpoint."""
        segment = self._segment
        while segment is None or segment.end <= self._time:
            solver = self._solver
            assert solver is not None
            if solver.status != 'running':
                self._restart(solver.t, solver.y, p)
                solver = self._solver
                assert solver is not None
            start, before = solver.t, solver.y.copy()
            message = solver.step()
            if solver.status == 'failed':
                largest = float(np.max(np.abs(solver.y))) if solver.y.size else 0.0
                if not np.isfinite(largest) or largest > _BLOW_UP:
                    error_msg = (
                        f'The solution blows up near t = {start}: the largest state reaches {largest:.3g}, and the '
                        f'{self.method_} integrator cannot follow it. Check the laws for growth without bound, or end '
                        'the run before it.'
                    )
                    raise SolverError(error_msg)
                error_msg = f'The {self.method_} integrator failed at t = {start}: {message}'
                raise SolverError(error_msg)
            if solver.t == start and solver.t_bound - start <= 2 * _RESTART * max(1.0, abs(start)):
                self._restart(solver.t_bound, before, p)
                continue
            if solver.t == start or not np.all(np.isfinite(solver.y)):
                error_msg = (
                    f'The {self.method_} integrator failed at t = {start}: its step makes no progress or the state is '
                    'not finite. Check the system for a solution that blows up.'
                )
                raise SolverError(error_msg)
            if self._stalls(start, solver.t):
                self._leave_lsoda(solver.t, solver.y, p)
            segment = _Segment(start, solver.t, before, solver.y.copy(), solver.dense_output())
            if self.history_ is not None:
                self.history_.add(start, solver.t, segment.dense)
            self._segment = segment
        return segment

    def _stalls(self, start: float, end: float) -> bool:
        """Check whether LSODA has taken many consecutive steps too short to advance the run, or too many in a frame.

        Args:
            start:
                The time at the start of the step.

            end:
                The time at the end of the step.

        Returns:
            stalled:
                Whether LSODA should give way to another method.

        Raises:
            SolverError:
                Another integrator takes too many steps in one frame, as when a state slides along a discontinuity.
        """
        self._frame_steps += 1
        if self.method_ != 'LSODA':
            if self._frame_steps >= _MAX_FRAME_STEPS:
                error_msg = (
                    f'The {self.method_} integrator took {_MAX_FRAME_STEPS} steps in one frame at t = {end}, as when a '
                    'state slides along a discontinuity of the drift, which no adaptive integrator crosses. Smooth the '
                    'switch of the law, or use a fixed step with RK4Solver.'
                )
                raise SolverError(error_msg)
            return False
        if self._frame_steps >= _MAX_FRAME_STEPS:
            return True
        if end - start >= _STALL_STEP * max(1.0, abs(end)):
            self._stalled = 0
            return False
        self._stalled += 1
        return self._stalled >= _STALL_STEPS

    def _leave_lsoda(self, t: float, x: Array, p: Array) -> None:
        """Continue with `RK45` after LSODA stalls on a default method, or raise the named error on an explicit one."""
        cause = (
            f'{_MAX_FRAME_STEPS} steps in one frame'
            if self._frame_steps >= _MAX_FRAME_STEPS
            else f'{_STALL_STEPS} consecutive steps shorter than {_STALL_STEP:g} of the time'
        )
        if self.method != 'auto':
            error_msg = (
                f'The LSODA integrator stalled at t = {t} with {cause}, as when a state rests on a discontinuity of '
                "the drift or slides along one. Use ScipyODESolver(method='RK45'), which needs no Jacobian."
            )
            raise SolverError(error_msg)
        warning_msg = (
            f'LSODA stalled at t = {t} with {cause}, as when a state rests on a discontinuity of the drift or slides '
            'along one, so the run continues with RK45.'
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=5)
        self.method_, self.jacobian_options_, self._exact, self._stalled = 'RK45', {}, False, 0
        self._frame_steps = 0
        self._restart(t, x, p)

    def _first_change(self, dense: Callable[[float], Array], end: float, p: Array) -> float:
        """Return the first time of a step, up to its end, at which the boundary laws change the state, by bisection."""

        def changed(time: float) -> bool:
            state = np.asarray(dense(time), dtype=float)
            return not np.array_equal(self._bounded(time, state, p), state)

        low, high = self._time, end
        if changed(low):
            return end
        while high - low > _RESTART * max(1.0, abs(high)):
            middle = (low + high) / 2
            low, high = (low, middle) if changed(middle) else (middle, high)
        return high

    def _to_event(self, p: Array, until: float) -> None:
        """Integrate the current step up to a time or to its first event, and apply the event or the boundary.

        The integrator restarts where a boundary law first changes the state. When that happens at once after a restart,
        the state rests on the boundary while the drift pushes it out, as for a clipped state under a constant inflow,
        so the step runs to its end and the boundary projects the state there instead.
        """
        segment = self._next_segment(p)
        end = min(segment.end, until)
        after = segment.after if end == segment.end else np.asarray(segment.dense(end), dtype=float)
        hit = (
            self.locator_.crossing(self._time, end, self._state, after, p=p, interpolant=segment.dense)
            if self.problem_.events
            else None
        )
        if hit is not None:
            time, index = hit
            state = self.locator_.apply(time, index, np.asarray(segment.dense(time), dtype=float), p)
            self._restart(time, self._bounded(time, state, p), p)
            return
        if np.array_equal(self._bounded(end, after, p), after):
            self._time, self._state = end, after
            return
        time = self._first_change(segment.dense, end, p)
        if time - self._time <= _RESTING_SPAN * max(1.0, abs(time)):
            time = end
        self._restart(time, self._bounded(time, np.asarray(segment.dense(time), dtype=float), p), p)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Integrate the drift from time `t` to time `until`, continuing the integrator of the previous call.

        The integrator restarts when the parameters change, when the state or time is not the one it returned, at
        breakpoints, at events, and when a boundary law changes the state.

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
        problem = self.problem_
        if until <= t or x.size == 0 or problem.drift is None:
            return x
        if problem.delays and self.history_ is None:
            self.history_ = History(problem, t, x, extrapolate=True)
        self._until = until
        self._frame_steps = 0
        self._choose_for_span(until - t)
        self._warn_stiff(t, until)
        if not self._continues(t, x, p):
            self._frame = until - t
            self._restart(t, x, p)
        while self._time < until - _RESTART * max(1.0, abs(until)):
            self._to_event(p, until)
            if self.locator_.stopped_at is not None:
                return self._state.copy()
        return self._state.copy()

    @staticmethod
    def _lsoda_first_step(
        field: Callable[[float, Array], Array],
        t: float,
        x: Array,
        *,
        end: float,
        max_step: float,
        rtol: float,
        atol: float,
    ) -> float | None:
        """Return the first step of LSODA when the drift vanishes at the start, or `None` to let LSODA choose.

        LSODA guesses its first step from the distance to the end when the drift vanishes at the start, and without a
        largest step it fails before its first step on a far end. The step then comes from the drift's change over a
        trial step, as in SciPy's other solvers (Hairer, Norsett, and Wanner, II.4). Elsewhere LSODA's own guess is
        kept, since a trial step across a discontinuous drift, such as stick-slip friction, gives needlessly small
        steps.
        """
        span = abs(end - t)
        with np.errstate(all='ignore'):
            drift = field(t, x)
            scale = atol + np.abs(x) * rtol
            drift_norm = np.sqrt(np.mean((drift / scale) ** 2))
            if drift_norm >= _SMALL_NORM:
                return None
            trial = min(_FALLBACK_STEP, span)
            change = np.sqrt(np.mean(((field(t + trial, x + trial * drift) - drift) / scale) ** 2)) / trial
            largest = max(drift_norm, change)
            second = max(_FALLBACK_STEP, trial * 1e-3) if largest <= _TINY_NORM else (0.01 / largest) ** 0.5
            step = min(100 * trial, second, span, max_step)
        return float(step) if math.isfinite(step) and step > 0 else None
