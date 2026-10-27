"""Integrate ordinary differential equations with adaptive SciPy steps, Runge-Kutta steps, or velocity Verlet steps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, OdeSolver, Radau

from ...core.errors import SolverError
from ...core.laws import (
    BoundaryLaw,
    CrossingLaw,
    DestructionLaw,
    EncounterLaw,
    EquationLaw,
    FluxLaw,
    JumpLaw,
    ProliferationLaw,
    RateLaw,
    TransferLaw,
)
from ...core.spaces import CayleySpace, ContinuousSpace, ContinuousTime, LatticeSpace, MeshSpace, NetworkSpace
from ...core.types import Vector
from ..problems import Form, OrderKind, Problem, SupportSpec, VectorFunction, components_of, step_count
from ._hybrid import (
    RESTART,
    History,
    Interpolant,
    advance_fixed_steps,
    find_left_limit,
    find_smallest_delay,
    hermite,
    is_hybrid,
    open_history,
    reject_hybrid,
    with_delays,
)
from ._stepper import Solver

METHODS = {'RK45': RK45, 'RK23': RK23, 'DOP853': DOP853, 'Radau': Radau, 'BDF': BDF, 'LSODA': LSODA}
IMPLICIT = frozenset({'Radau', 'BDF', 'LSODA'})
JACOBIANS = ('auto', 'exact', 'dense')
DENSITY_LIMIT = 0.5
BAND_FRACTION = 0.1
STIFF_SPAN = 1000
EXPLICIT_SPAN = 100
DISCONTINUITY_ORDERS = 3
STEPS_PER_FRAME = 100
LARGEST_TIME = sys.float_info.max
LSODA_LARGEST_SIZE = 4096
STALL_STEP = 1e-9
RESTING_SPAN = 1e-9
STALL_STEPS = 1000
MAX_FRAME_STEPS = 50_000
BLOW_UP = 1e12
SMALL_NORM = 1e-5
TINY_NORM = 1e-15
FALLBACK_STEP = 1e-6


def _lsoda_first_step(
    field: Callable[[float, Vector], Vector],
    t: float,
    x: Vector,
    *,
    end: float,
    max_step: float,
    rtol: float,
    atol: float,
) -> float | None:
    """Return the first step of LSODA when the drift vanishes at the start, or `None` to let LSODA choose.

    LSODA guesses its first step from the distance to the end when the drift vanishes at the start, and without a
    largest step it fails before its first step on a far end. The step then comes from the drift's change over a trial
    step, as in SciPy's other solvers (Hairer, Norsett, and Wanner, II.4). Elsewhere LSODA's own guess is kept, since a
    trial step across a discontinuous drift, such as stick-slip friction, gives needlessly small steps.
    """
    span = abs(end - t)
    with np.errstate(all='ignore'):
        drift = field(t, x)
        scale = atol + np.abs(x) * rtol
        drift_norm = np.sqrt(np.mean((drift / scale) ** 2))
        if drift_norm >= SMALL_NORM:
            return None
        trial = min(FALLBACK_STEP, span)
        change = np.sqrt(np.mean(((field(t + trial, x + trial * drift) - drift) / scale) ** 2)) / trial
        largest = max(drift_norm, change)
        second = max(FALLBACK_STEP, trial * 1e-3) if largest <= TINY_NORM else (0.01 / largest) ** 0.5
        step = min(100 * trial, second, span, max_step)
    return float(step) if math.isfinite(step) and step > 0 else None


def _position_pairs(problem: Problem) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the flat indices of the position components and of their velocities, in matching order."""
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
            f'VelocityVerletSolver integrates positions and velocities, with dx/dt = v, but {", ".join(missing)} are '
            'neither. Use RK4Solver or ScipyODESolver for this system.'
        )
        raise SolverError(error_msg)
    return position_indices, velocity_indices


def rk4_step(field: Callable[[float, Vector], Vector], time: float, state: Vector, step: float | Vector) -> Vector:
    """Return the state after one classical Runge-Kutta step, with one step length or one per state component.

    The step spans the longest of the step lengths in time.

    Args:
        field:
            The drift `field(t, x)`.

        time:
            The time at the start of the step.

        state:
            The state at the start of the step.

        step:
            The step length, or one step length per state component.

    Returns:
        state:
            The state at the end of the step, before the boundary laws.
    """
    span = float(np.max(step))
    k1 = field(time, state)
    k2 = field(time + span / 2, state + step / 2 * k1)
    k3 = field(time + span / 2, state + step / 2 * k2)
    k4 = field(time + span, state + step * k3)
    return state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass(frozen=True)
class _Segment:
    """Accepted step of an integrator, with its states at both ends and its dense output."""

    start: float
    end: float
    before: Vector
    after: Vector
    dense: Callable[[float], Vector]


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
        >>> from skcomplex.engine.solvers import ScipyODESolver
        >>> problem = compile_system(
        ...     System(
        ...         elements=[PopulationElement('a', n=1)],
        ...         interactions=[ExponentialGrowthInteraction('g', target='a', r=1)],
        ...     ),
        ... )
        >>> stepper = ScipyODESolver(rtol=1e-8).prepare(problem, np.random.default_rng(0))
        >>> x = stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0)
        >>> round(float(x[0]), 4)
        2.7183
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={RateLaw, FluxLaw, EquationLaw, CrossingLaw, BoundaryLaw, TransferLaw, EncounterLaw, ProliferationLaw, DestructionLaw},
        
        
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.ODE

    def __init__(self, method: str = 'auto', rtol: float = 1e-6, atol: float = 1e-9, jacobian: str = 'auto') -> None:
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.jacobian = jacobian

    def _prepare(self, problem: Problem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the stepper for a problem, with no integrator yet.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator, unused by this deterministic stepper.

            until:
                The end time of the run.

        Raises:
            SolverError:
                The problem is not an ODE, the method or the Jacobian option is unknown, or `LSODA` would store a dense
                matrix larger than its limit.

            ValidationError:
                The exact Jacobian is asked for laws without a symbolic form.
        """
        super()._prepare(problem, rng, until)
        if self.method != 'auto' and self.method not in METHODS:
            error_msg = f'Unknown method {self.method!r}. Use one of auto, {", ".join(METHODS)}.'
            raise SolverError(error_msg)
        if self.jacobian not in JACOBIANS:
            error_msg = f'Unknown Jacobian option {self.jacobian!r}. Use one of {", ".join(map(repr, JACOBIANS))}.'
            raise SolverError(error_msg)
        self.jacobian_pattern_ = None
        self.jacobian_options_: dict[str, Any] = {}
        self._exact = False
        self._warned = False
        self._stalled = 0
        self._frame_steps = 0
        self._choosing = False
        self.method_ = 'LSODA' if self.method == 'auto' else self.method
        if self.method_ in IMPLICIT and self.jacobian != 'dense' and problem.jacobian is not None:
            self._prepare_jacobian(problem)
        if self.method_ == 'LSODA' and not self._fits_lsoda(problem.initial_state.size):
            if self.method != 'auto':
                error_msg = (
                    f'LSODA would store a dense {problem.initial_state.size} x {problem.initial_state.size} matrix, '
                    f'more than the limit of {LSODA_LARGEST_SIZE} states without a narrow band. Use '
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
        return size <= LSODA_LARGEST_SIZE or 'lband' in self.jacobian_options_

    def _prepare_jacobian(self, problem: Problem) -> None:
        """Derive the Jacobian structure an implicit method receives, or check that the exact Jacobian exists."""
        jacobian = problem.jacobian
        assert jacobian is not None
        if self.jacobian == 'exact':
            jacobian.validate()
            self._exact = True
            self.jacobian_pattern_ = jacobian.pattern()
            return
        pattern = jacobian.pattern()
        size = problem.initial_state.size
        if pattern is None or pattern.nnz > DENSITY_LIMIT * size * size:
            return
        self.jacobian_pattern_ = pattern
        if self.method_ != 'LSODA':
            self.jacobian_options_ = {'jac_sparsity': pattern}
            return
        rows, columns = pattern.nonzero()
        lower = int(np.max(rows - columns, initial=0))
        upper = int(np.max(columns - rows, initial=0))
        if max(lower, upper) <= BAND_FRACTION * size:
            self.jacobian_options_ = {'lband': lower, 'uband': upper}

    def _jacobian_options(self, history: History | None, p: Vector) -> dict[str, Any]:
        """Return the keyword arguments that pass the Jacobian structure to the integrator."""
        if not self._exact:
            return self.jacobian_options_
        problem = self.problem_
        jacobian = problem.jacobian
        assert jacobian is not None
        dense = self.method_ == 'LSODA'

        def exact(time: float, state: Vector) -> Any:  # noqa: ANN401
            matrix = jacobian.evaluate(time, state, with_delays(problem, history, time, state, p))
            return matrix.toarray() if dense else matrix

        return {'jac': exact}

    def _choose_for_span(self, span: float) -> None:
        """Choose RK45 or BDF for a large diffusive system by how many stable steps the first call spans."""
        if not self._choosing:
            return
        self._choosing = False
        limit = self.problem_.stable_step
        if limit is not None and span > EXPLICIT_SPAN * limit:
            self.method_ = 'BDF'
            self._prepare_jacobian(self.problem_)

    def _warn_stiff(self, t: float, until: float) -> None:
        """Warn once when an explicit method spans many stable steps of the diffusion of a field in one call."""
        limit = self.problem_.stable_step
        if self._warned or self.method_ in IMPLICIT or limit is None or until - t <= STIFF_SPAN * limit:
            return
        self._warned = True
        warning_msg = (
            f'{self.method_} spans {until - t:.6g} in one call, more than {STIFF_SPAN} times the stability limit '
            f'{limit:.6g} of the diffusion of the fields, so its steps are limited by stability. Use '
            "ScipyODESolver(method='BDF') or ScipyODESolver(method='Radau'), which receive the sparsity pattern of the "
            "Jacobian."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=3)

    def _continues(self, t: float, x: Vector, p: Vector) -> bool:
        """Check whether a call continues the state, time, and parameters the integrator returned last."""
        return (
            self._solver is not None
            and abs(t - self._time) <= RESTART * max(1.0, abs(t))
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
            for order in range(1, DISCONTINUITY_ORDERS + 1)
            if history.start + order * delay.minimum > time
        ]
        return min(times, default=math.inf)

    def _restart(self, t: float, x: Vector, p: Vector) -> None:
        """Start a new integrator at a time and state, bounded by the next breakpoint."""
        problem = self.problem_
        drift = problem.drift
        history = self.history_
        if history is not None:
            history.truncate(t)
        after = t + RESTART * max(1.0, abs(t))
        later = problem.breakpoints.next_after(after)
        if problem.breakpoints.unresolved:
            later = self._until if later is None else min(later, self._until)
        bound = min(later if later is not None else math.inf, self._next_discontinuity(after))
        ceiling = find_left_limit(bound) if math.isfinite(bound) and later == bound else math.inf

        def field(time: float, state: Vector) -> Vector:
            time = min(time, ceiling)
            return drift(time, state, with_delays(problem, history, time, state, p))  # type: ignore[misc]

        delay = find_smallest_delay(problem)
        end = min(bound, LARGEST_TIME)
        max_step = max(delay, self._frame / STEPS_PER_FRAME)
        first_step = min(delay / 2, bound - t) if problem.delays else None
        if first_step is None and self.method_ == 'LSODA':
            first_step = _lsoda_first_step(field, t, x, end=end, max_step=max_step, rtol=self.rtol, atol=self.atol)
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

    def _next_segment(self, p: Vector) -> _Segment:
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
                if not np.isfinite(largest) or largest > BLOW_UP:
                    error_msg = (
                        f'The solution blows up near t = {start}: the largest state reaches {largest:.3g}, and the '
                        f'{self.method_} integrator cannot follow it. Check the laws for growth without bound, or end '
                        'the run before it.'
                    )
                    raise SolverError(error_msg)
                error_msg = f'The {self.method_} integrator failed at t = {start}: {message}'
                raise SolverError(error_msg)
            if solver.t == start and solver.t_bound - start <= 2 * RESTART * max(1.0, abs(start)):
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
            if self._frame_steps >= MAX_FRAME_STEPS:
                error_msg = (
                    f'The {self.method_} integrator took {MAX_FRAME_STEPS} steps in one frame at t = {end}, as when a '
                    'state slides along a discontinuity of the drift, which no adaptive integrator crosses. Smooth the '
                    'switch of the law, or use a fixed step with RK4Solver.'
                )
                raise SolverError(error_msg)
            return False
        if self._frame_steps >= MAX_FRAME_STEPS:
            return True
        if end - start >= STALL_STEP * max(1.0, abs(end)):
            self._stalled = 0
            return False
        self._stalled += 1
        return self._stalled >= STALL_STEPS

    def _leave_lsoda(self, t: float, x: Vector, p: Vector) -> None:
        """Continue with `RK45` after LSODA stalls on a default method, or raise the named error on an explicit one."""
        cause = (
            f'{MAX_FRAME_STEPS} steps in one frame'
            if self._frame_steps >= MAX_FRAME_STEPS
            else f'{STALL_STEPS} consecutive steps shorter than {STALL_STEP:g} of the time'
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

    def _first_change(self, dense: Callable[[float], Vector], end: float, p: Vector) -> float:
        """Return the first time of a step, up to its end, at which the boundary laws change the state, by bisection."""

        def changed(time: float) -> bool:
            state = np.asarray(dense(time), dtype=float)
            return not np.array_equal(self._bounded(time, state, p), state)

        low, high = self._time, end
        if changed(low):
            return end
        while high - low > RESTART * max(1.0, abs(high)):
            middle = (low + high) / 2
            low, high = (low, middle) if changed(middle) else (middle, high)
        return high

    def _to_event(self, p: Vector, until: float) -> None:
        """Integrate the current step up to a time or to its first event, and apply the event or the boundary.

        The integrator restarts where a boundary law first changes the state. When that happens at once after a restart,
        the state rests on the boundary while the drift pushes it out, as for a clipped stock under a constant inflow,
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
        if time - self._time <= RESTING_SPAN * max(1.0, abs(time)):
            time = end
        self._restart(time, self._bounded(time, np.asarray(segment.dense(time), dtype=float), p), p)

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
        while self._time < until - RESTART * max(1.0, abs(until)):
            self._to_event(p, until)
            if self.locator_.stopped_at is not None:
                return self._state.copy()
        return self._state.copy()


class RK4Solver(Solver):
    """Classical fourth-order Runge-Kutta integrator of ordinary differential equations with a fixed internal step.

    With the Numba backend every internal step of a frame runs in compiled code.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationElement
        >>> from skcomplex.engine.solvers import RK4Solver
        >>> problem = compile_system(
        ...     System(
        ...         elements=[PopulationElement('a', n=1)],
        ...         interactions=[ExponentialGrowthInteraction('g', target='a', r=1)],
        ...     ),
        ... )
        >>> stepper = RK4Solver(dt=0.001).prepare(problem, np.random.default_rng(0))
        >>> round(float(stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0)[0]), 10)
        2.7182818285
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={
            JumpLaw,
            RateLaw, FluxLaw,
            EquationLaw,
            CrossingLaw,
            BoundaryLaw,
            TransferLaw,
            EncounterLaw, ProliferationLaw, DestructionLaw,
        },
        
        
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.ODE
    renews: ClassVar[bool] = True

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def _step(self, time: float, state: Vector, p: Vector, lengths: float | Vector) -> Vector:
        """Return the state after one Runge-Kutta step and the boundary laws, with one step length or one per state."""
        drift = cast('VectorFunction', self.problem_.drift)

        def field(time: float, state: Vector) -> Vector:
            return drift(time, state, p)

        return self._bounded(time + float(np.max(lengths)), rk4_step(field, time, state, lengths), p)

    def _advance_hybrid(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Integrate with fixed RK4 steps, interpolating every step for delays and locating events in it."""
        problem = self.problem_
        drift = cast('VectorFunction', problem.drift)
        history = self.history_ = open_history(problem, self.history_, t, x)

        def take(time: float, state: Vector, step: float, ceiling: float) -> tuple[Vector, Interpolant]:
            def field(time: float, state: Vector) -> Vector:
                time = min(time, ceiling)
                return drift(time, state, with_delays(problem, history, time, state, p))

            following = rk4_step(field, time, state, step)
            interpolant = hermite(
                time,
                state,
                field(time, state),
                t1=time + step,
                x1=following,
                f1=field(time + step, following),
            )
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
        if self.problem_.renewals:
            return self._advance_renewals(t, x, p, until, dt=self.dt, step=self._step)
        if is_hybrid(self.problem_, t, until):
            return self._advance_hybrid(t, x, p, until)
        accelerated = self.problem_.accelerated
        if accelerated is not None and accelerated.rk4 is not None:
            return accelerated.rk4(t, x, p, until, self.dt)
        n_steps = step_count(t, until, self.dt)
        step = (until - t) / n_steps
        state = x.copy()
        for index in range(n_steps):
            state = self._step(t + index * step, state, p, step)
        return state


class VelocityVerletSolver(Solver):
    """Velocity Verlet integrator of particle systems, whose positions move with their velocities.

    Every step kicks the velocities by half a step, moves the positions, applies the boundary laws, and kicks the
    velocities again with the accelerations at the new positions, so a step costs one drift evaluation. It conserves
    energy over long runs of conservative forces. With the Numba backend every internal step of a frame runs in
    compiled code.

    Args:
        dt:
            The largest internal time step.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.spaces import ContinuousSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.core.elements import ParticleElement
        >>> from skcomplex.models.physics.realizations.particles import ConstantForceInteraction
        >>> from skcomplex.engine.solvers import VelocityVerletSolver
        >>> problem = compile_system(
        ...     System(
        ...         elements=[ParticleElement('ball')],
        ...         interactions=[ConstantForceInteraction('push', target='ball', F=2.0)],
        ...         space=ContinuousSpace(n_dim=1),
        ...     ),
        ... )
        >>> stepper = VelocityVerletSolver(dt=0.1).prepare(problem, np.random.default_rng(0))
        >>> stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0).round(10).tolist()
        [1.0, 2.0]
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={ContinuousTime},
        space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
        laws={RateLaw, FluxLaw, EquationLaw, BoundaryLaw},
        
        
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.ODE

    def __init__(self, dt: float = 1e-3) -> None:
        self.dt = dt

    def _prepare(self, problem: Problem, rng: np.random.Generator, until: float = math.inf) -> None:
        """Prepare the stepper and find the position and velocity components of the problem.

        Args:
            problem:
                The compiled problem.

            rng:
                The random number generator, unused by this deterministic stepper.

            until:
                The end time of the run.

        Raises:
            SolverError:
                The problem is not an ODE, or has states that are neither positions nor velocities.
        """
        super()._prepare(problem, rng, until)
        reject_hybrid(problem, type(self).__name__)
        self.positions_, self.velocities_ = _position_pairs(problem)

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
