"""Run compiled systems frame by frame, offline, in real time, or in the background."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ...base import BaseConfig
from ...core.system import System
from ...errors import ParameterError, SimulationError, ValidationError
from ...specs import (
    AgentStateSpec,
    BaseProblemSpec,
    EntrySpec,
    FrameSpec,
    ObservableSpec,
    RunStateSpec,
    StepSpec,
    StructureStateSpec,
)
from ...types import Array, FormKind
from ..backends import BaseBackend
from ..compile import ObservableEvaluator, PassCompiler, compile_system
from ..solvers import BaseSolver, CoSimulationSolver
from ..solvers._defaults import DEFAULT_SOLVERS
from ._as_fast_as_possible import AsFastAsPossibleClock
from ._base import BaseClock, BaseObserver
from ._invariants import InvariantObserver
from ._recorder import RecorderObserver
from ._trajectories import Trajectory

_DEFAULT_DT = 0.01
_MAX_FRAMES = 1000
_FRAME_MANTISSAS = (1, 2, 5)
TIME_TOLERANCE = 1e-9


class Simulation(BaseConfig):
    """Simulation of a system with the solver of its form.

    Each frame advances simulated time by `dt` and is handed to every observer. Parameters can change between
    frames through `update_parameters`, also while a background run started with `start` is going. Values set
    between runs stay in force for the later runs of the simulation, and `extend` continues a run from its last
    frame.

    A simulation composes the pipeline of the engine and follows its common API: `check()` validates the system,
    `prepare()` compiles it and chooses the solver, and `run(until)` prepares and runs from time 0 until a time. A run
    prepares the solver for its end time and every observer on the problem before the first frame, runs every observer
    at every frame, and runs it with `final=True` at the end.

    Args:
        system:
            The system to simulate, or a problem already compiled from one, which skips compiling for repeated runs
            that change only parameters or initial values.

        solver:
            The solver, or `None` for the default solver of the system's form, `CoSimulationSolver` for a system that
            runs as partitions.
            The default ODE solver is `ScipyODESolver` with `LSODA`, which switches to BDF methods while the system is
            stiff.

        backend:
            The backend that compiles the system, or `None` for `NumbaBackend` on systems with at least 1000 members and
            `NumpyBackend` on the others.

        dt:
            The simulated time between frames. It defaults to 1 for discrete maps and agent-based models. Otherwise
            it defaults to 0.01, raised to the smallest of 0.02, 0.05, 0.1, 0.2, and so on, in steps of 1, 2, and 5
            times a power of ten, that keeps a run at most 1000 frames long.

        clock:
            The clock that paces frames against wall time. It defaults to running as fast as possible.

        observers:
            The observers that receive every frame.

        record:
            Whether to record the frames as the trajectory `trajectory_`.

        random_state:
            The seed or generator of stochastic solvers.

        checked:
            Whether to check after every frame that every state lies in its domain, such as an alphabet of labels,
            and raise an error naming the state when it does not.

        schedule:
            A mapping from dotted parameter name to a function of time, such as a ramp of a control parameter. At every
            frame the parameter takes the value of its function at the frame's time and keeps it until the next frame.
            An adaptive parameter, such as a bounded control input, can be scheduled too, and the trajectory records
            its scheduled values.

        observables:
            The observables to compute and record at every frame, labelled by their path from the root, such as
            `village/mean_wealth` for an observable of the nested system `village`.

        invariants:
            The tolerance of every observable that must stay constant, by the path of the observable. At every frame
            the run checks that each stays within its tolerance of its value at the first frame, and raises a
            `ConservationError` that names the observable and the first frame that breaks it.

        initial_state:
            Initial values that replace those of the elements, by dotted state name, such as `{'rabbits.n': 20}`.
            The whole state of a quantum system is `<system label>.state`, which takes a vector, a density matrix, or
            `'ground'` for the ground state.

    Attributes:
        problem_:
            The compiled problem.

        stepper_:
            The stepper that advances the problem.

        time_:
            The simulated time of the last frame.

        state_:
            The state vector of the last frame.

        parameters_:
            The current parameter vector.

    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import LogisticGrowthInteraction, PopulationElement
        >>> from skcomplex.engine.runtime import Simulation
        >>> system = System(
        ...     elements=[PopulationElement('rabbits', n=10)],
        ...     interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> simulation = Simulation(system, dt=1.0).check().run(until=20)
        >>> round(simulation.trajectory_.values('rabbits.n')[-1])
        100
    """

    def __init__(
        self,
        system: System | BaseProblemSpec,
        *,
        solver: BaseSolver | None = None,
        backend: BaseBackend | None = None,
        dt: float | None = None,
        clock: BaseClock | None = None,
        observers: Sequence[BaseObserver] = (),
        record: bool = True,
        random_state: int | np.random.Generator | None = None,
        checked: bool = False,
        schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None = None,
        observables: Sequence[ObservableSpec] = (),
        invariants: Mapping[str, float] | None = None,
        initial_state: Mapping[str, object] | None = None,
    ) -> None:
        self.system = system
        self.solver = solver
        self.backend = backend
        self.dt = dt
        self.clock = clock
        self.observers = observers
        self.record = record
        self.random_state = random_state
        self.checked = checked
        self.schedule = schedule
        self.observables = observables
        self.invariants = invariants
        self.initial_state = initial_state

    def _check_dt(self, problem: BaseProblemSpec, until: float) -> float:
        """Return the time between frames of a run until a time, after validating it for the problem's form."""
        forms = {problem.form, *(item.form for item in self._partition_problems(problem))}
        stepwise = bool(
            forms & {FormKind.DISCRETE, FormKind.AGENT, FormKind.LATTICE, FormKind.MONTE_CARLO, FormKind.REWRITING},
        )
        default = 1.0 if stepwise else self._default_dt(until)
        dt = default if self.dt is None else self.dt
        if not EntrySpec._contains(RealSet.open(0, oo), dt):
            error_msg = f'dt = {dt!r} must be a positive number.'
            raise ValidationError(error_msg)
        if stepwise and not EntrySpec._contains(ZZ, dt):
            error_msg = f'dt = {dt!r} must be a positive integer for a {problem.form.value}.'
            raise ValidationError(error_msg)
        return float(dt)

    def _previous_parameters(self) -> tuple[list[str], Array] | None:
        """Return the parameter names and values of the previous run of the same system, or `None` without one."""
        if getattr(self, '_prepared_system', None) is not self.system or not hasattr(self, 'parameters_'):
            return None
        problem: BaseProblemSpec = self.problem_
        return problem.parameters.names, self.parameters_.copy()

    def _check(self) -> None:
        """Validate the system against the solver, the backend, and the observables, unless it is a compiled problem.

        Raises:
            ValidationError:
                The system is not a system or a compiled problem, such as a model name, or it cannot be compiled or
                run, with every problem found.
        """
        if not isinstance(self.system, System | BaseProblemSpec):
            error_msg = (
                f'A simulation takes a system or a compiled problem, not {self.system!r}. Create a model of the '
                'library with `skcomplex.models.create`, or simulate it by name with `skcomplex.models.run`.'
            )
            raise ValidationError(error_msg)
        if not isinstance(self.system, BaseProblemSpec):
            compiler = PassCompiler(backend=self.backend, solver=self.solver, observables=self.observables)
            compiler.check().prepare(self.system).report_.check()

    def _prepare(self, context: None = None) -> None:
        """Compile the system, choose the solver, and gather the clock and the observers of a run."""
        previous = self._previous_parameters()
        initial = dict(self.initial_state or {})
        self.problem_ = (
            self.system.with_initial_state(initial)
            if isinstance(self.system, BaseProblemSpec)
            else compile_system(self.system, self.backend, self.observables, solver=self.solver, initial_state=initial)
        )
        self._prepared_system = self.system
        self._adaptive = self.problem_.adaptive_parameters
        self.stepper_ = self.solver if self.solver is not None else self._default_solver(self.problem_.form)
        self.observables_ = ObservableEvaluator(self.problem_).prepare()
        self.clock_ = self.clock if self.clock is not None else AsFastAsPossibleClock()
        self.recorder_ = RecorderObserver() if self.record else None
        monitors = [InvariantObserver(self.invariants)] if self.invariants else []
        self.observers_ = [*monitors, *self.observers, *([self.recorder_] if self.recorder_ is not None else [])]
        self.parameters_ = self.problem_.initial_parameters.copy()
        if previous is not None and previous[0] == self.problem_.parameters.names:
            self.parameters_ = previous[1] if previous[1].shape == self.parameters_.shape else self.parameters_
        self._pending: queue.Queue[tuple[slice, Array]] = queue.Queue()
        self._stop_requested = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _start(self, until: float) -> None:
        """Prepare the solver and the observers for a run from time 0 until a time.

        Args:
            until:
                The simulated time the run stops at.
        """
        self.dt_ = self._check_dt(self.problem_, until)
        rng = np.random.default_rng(self.random_state)
        self.stepper_.set_params(random_state=rng, until=until).check().prepare(self.problem_)
        self.time_ = 0.0
        self.state_ = self.stepper_.initial_state_.copy()
        self._step = 0
        for observer in self.observers_:
            observer.check().prepare(self.problem_)

    def _emit(self, step: int) -> None:
        """Hand the current frame, with the events since the previous one, to every observer."""
        external = self.stepper_.snapshot_
        agents, reporters = (external.groups, external.reporters) if isinstance(external, AgentStateSpec) else ({}, {})
        structures = external.structures if isinstance(external, StructureStateSpec) else {}
        reporters = external.reporters if isinstance(external, StructureStateSpec) else reporters
        observables = self.observables_.evaluate(self.time_, self.state_, self.parameters_, self.stepper_.agents_.get)
        frame = FrameSpec(
            step,
            self.time_,
            self.state_.copy(),
            self.parameters_.copy(),
            self.problem_,
            agents,
            reporters,
            observables,
            tuple(self.stepper_.events_),
            structures,
        )
        for observer in self.observers_:
            observer.run(frame)

    def _apply_pending(self) -> None:
        """Apply the parameter changes queued since the last frame."""
        while not self._pending.empty():
            indices, values = self._pending.get_nowait()
            self.parameters_[indices] = values

    def _loop(self, until: float) -> None:
        """Advance frame by frame from the current time until `until` or until a stop is requested.

        The stepper runs once per frame, and its last run, at the last frame or after a stop, releases what it acquired,
        such as an external process.
        """
        origin, first = self.time_, self._step
        self.clock_.check().prepare()
        self._apply_schedule()
        if first == 0:
            self._emit(0)
        released = False
        try:
            while self.time_ < until - TIME_TOLERANCE and not self._stop_requested.is_set():
                self._running.wait()
                if self._stop_requested.is_set():
                    break
                self._apply_pending()
                self._step += 1
                target = min(self._step * self.dt_, until)
                released = target >= until - TIME_TOLERANCE
                self.state_ = self.stepper_.run(
                    StepSpec(self.time_, self.state_, self.parameters_, target),
                    final=released,
                )
                stopped_at = self.stepper_.stopped_at_
                if stopped_at is not None:
                    target = stopped_at
                self.time_ = target
                self._apply_schedule()
                if self.checked:
                    self._check_domains()
                self._emit(self._step)
                if stopped_at is not None:
                    break
                self.clock_.run(self.time_ - origin)
        finally:
            if not released:
                self.stepper_.run(StepSpec(self.time_, self.state_, self.parameters_, self.time_), final=True)
            for observer in self.observers_:
                observer.run(final=True)

    def _apply_schedule(self) -> None:
        """Set every scheduled parameter and adaptive parameter to the value of its function at the current time."""
        for name, function in (self.schedule or {}).items():
            if name in self._adaptive:
                entry, indices = self.problem_.states.locate(name)
                self.state_[indices] = self._check_value(name, entry, function(self.time_))
                continue
            layout = self.problem_.parameters
            try:
                entry, indices = layout.locate(name)
            except KeyError as error:
                names = ', '.join([*layout.names, *self._adaptive])
                error_msg = f'No parameter named {name!r} to schedule. Available names are: {names}.'
                raise ParameterError(error_msg) from error
            self.parameters_[indices] = self._check_value(name, entry, function(self.time_))

    def _run_in_background(self, until: float) -> None:
        """Run the loop and keep any error for `wait` to raise."""
        try:
            self._loop(until)
        except BaseException as error:  # noqa: BLE001
            self._error = error

    def _check_domains(self) -> None:
        """Raise an error when a state has left its domain."""
        for entry in self.problem_.states.entries:
            values = np.asarray(self.problem_.states.read(self.state_, entry.name))
            if not entry.contains(values):
                error_msg = (
                    f'{entry.name} took the values {np.round(values, 6).tolist()} at t = {self.time_}, outside its '
                    f'domain {entry.domain}. Check the laws that change it.'
                )
                raise ValidationError(error_msg)

    def _check_value(self, name: str, entry: EntrySpec, value: float | npt.ArrayLike) -> Array:
        """Return a parameter value broadcast to its entry after checking its domain and shape."""
        error_msg = (
            f'{name} = {value!r} is not a value of shape {entry.shape}, or of a shape that broadcasts to it, in the '
            f'domain {entry.domain}.'
        )
        if not entry.contains(value):
            raise ValidationError(error_msg)
        try:
            return entry.encode(value)
        except ValueError as error:
            raise ValidationError(error_msg) from error

    def _check_matrices(self, changes: list[tuple[str, slice, Array]]) -> None:
        """Raise an error when parameter changes would break a transition or emission matrix."""
        problems = (self.problem_, *self._partition_problems(self.problem_))
        matrices = [matrix for problem in problems for matrix in problem.matrices]
        if not matrices:
            return
        candidate = self.parameters_.copy()
        positions = np.arange(candidate.size)
        changed: dict[int, str] = {}
        for name, indices, value in changes:
            candidate[indices] = value
            changed.update(dict.fromkeys(positions[indices].tolist(), name))
        for matrix in matrices:
            names = sorted({changed[index] for index in matrix.reads if index in changed})
            violation = matrix.run(RunStateSpec(self.time_, self.state_, candidate)) if names else None
            if violation is not None:
                error_msg = (
                    f'Setting {", ".join(names)} would make {violation}, so the change is rejected and the parameters '
                    'keep their values. Choose values that keep every entry nonnegative and every row summing to 1.'
                )
                raise ParameterError(error_msg)

    def _check_prepared(self) -> None:
        """Raise an error when the simulation has not been run or started."""
        if not hasattr(self, 'problem_'):
            error_msg = 'The simulation has not been run or started. Call `run` or `start` first.'
            raise SimulationError(error_msg)

    def _default_solver(self, form: FormKind) -> BaseSolver:
        """Return the default solver of a form, `CoSimulationSolver` for a problem that runs as partitions."""
        return CoSimulationSolver() if form == FormKind.COUPLED else DEFAULT_SOLVERS[form]()

    def prepare(self) -> Self:
        """Compile the system of the checked simulation and choose the solver.

        Returns:
            self:
                The prepared simulation, with the compiled problem `problem_` and the solver `stepper_`.
        """
        return super().prepare()

    def run(self, until: float) -> Self:
        """Prepare the simulation and run it from time 0 to `until` in the calling thread.

        Args:
            until:
                The simulated time to stop at.

        Returns:
            simulation:
                The simulation itself.
        """
        return super().run(until)

    def _run(self, until: float) -> Self:
        """Prepare the simulation and run it from time 0 to `until` in the calling thread.

        The run starts from the initial state and keeps the parameter values of the previous run of the same system,
        such as values set with `update_parameters` between runs. `clone(simulation).run(until)` starts afresh.

        Args:
            until:
                The simulated time to stop at.

        Returns:
            simulation:
                The simulation itself.
        """
        self.prepare()
        self._start(until)
        self._loop(until)
        return self

    def extend(self, until: float) -> Self:
        """Continue a finished run from its last frame to `until` in the calling thread.

        The run keeps its state, parameters, stepper, clock, observers, and frame duration, applies pending parameter
        changes, and appends its frames to the trajectory. A solver that closes an external process at the end of a
        run, such as `LammpsSolver`, cannot continue it.

        Args:
            until:
                The simulated time to stop at, later than the time of the last frame.

        Returns:
            simulation:
                The simulation itself.

        Raises:
            SimulationError:
                The simulation has not run, is running in the background, ended at a terminal event, or `until` is not
                later than its time.

        Examples:
            >>> from skcomplex.core.system import System
            >>> from skcomplex.models.biology.ecology import LogisticGrowthInteraction, PopulationElement
            >>> from skcomplex.engine.runtime import Simulation
            >>> system = System(
            ...     elements=[PopulationElement('rabbits', n=10)],
            ...     interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=100)],
            ... )
            >>> simulation = Simulation(system, dt=1.0).check().run(until=5)
            >>> simulation.update_parameters({'growth.r': 0.0})
            >>> trajectory = simulation.extend(until=10).trajectory_
            >>> trajectory.times.tolist()
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            >>> bool(trajectory.values('rabbits.n')[5] == trajectory.values('rabbits.n')[10])
            True
        """
        self._check_prepared()
        if not hasattr(self, 'time_'):
            error_msg = 'The simulation has been prepared but not run. Call `run` before `extend`.'
            raise SimulationError(error_msg)
        if self._thread is not None and self._thread.is_alive():
            error_msg = 'The simulation is running in the background. Call `wait` or `stop` before `extend`.'
            raise SimulationError(error_msg)
        if not until > self.time_ + TIME_TOLERANCE:
            error_msg = f'Cannot extend the run to t = {until}, which is not later than its current time {self.time_}.'
            raise SimulationError(error_msg)
        if self.stepper_.stopped_at_ is not None:
            error_msg = f'A terminal event ended the run at t = {self.time_}, so it cannot be extended.'
            raise SimulationError(error_msg)
        self._stop_requested.clear()
        self._running.set()
        self._loop(until)
        return self

    def start(self, until: float = math.inf) -> Self:
        """Start the simulation from time 0 in a background thread.

        Args:
            until:
                The simulated time to stop at. The default runs until `stop` is called.

        Returns:
            simulation:
                The simulation itself.
        """
        self.prepare()
        self._start(until)
        self._thread = threading.Thread(target=self._run_in_background, args=(until,), daemon=True)
        self._thread.start()
        return self

    def pause(self) -> None:
        """Pause a background run before its next frame."""
        self._check_prepared()
        self._running.clear()

    def resume(self) -> None:
        """Resume a paused background run."""
        self._check_prepared()
        self._running.set()

    def wait(self, timeout: float | None = None) -> None:
        """Wait for a background run to finish and raise any error it met.

        Args:
            timeout:
                The longest time in seconds to wait, or `None` to wait until the run ends.

        Raises:
            BaseException:
                The error that ended the background run, if any.
        """
        self._check_prepared()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def stop(self) -> None:
        """Stop a background run after its current frame and wait for its thread to end."""
        self._check_prepared()
        self._stop_requested.set()
        self._running.set()
        self.wait()

    def update_parameters(self, values: Mapping[str, float | npt.ArrayLike]) -> None:
        """Set parameter values by dotted name, applied before the next frame.

        Args:
            values:
                A mapping from dotted parameter name, such as `growth.r`, or name of one component, such as
                `regulation.weights[1]`, to its new value.

        Raises:
            ParameterError:
                A name is not a parameter or a parameter component of the problem, or a value would make a
                transition or emission matrix negative or its rows not sum to 1.

            ValidationError:
                A value has the wrong shape or is outside the domain of its parameter.
        """
        self._check_prepared()
        layout = self.problem_.parameters
        changes = []
        for name, value in values.items():
            try:
                entry, indices = layout.locate(name)
            except KeyError as error:
                error_msg = f'No parameter named {name!r}. Available names are: {", ".join(layout.names)}.'
                raise ParameterError(error_msg) from error
            changes.append((name, indices, self._check_value(name, entry, value)))
        self._check_matrices(changes)
        for _, indices, value in changes:
            self._pending.put((indices, value))
        if self._thread is None or not self._thread.is_alive():
            self._apply_pending()

    @property
    def trajectory_(self) -> Trajectory:
        """Trajectory of the frames recorded so far."""
        self._check_prepared()
        if self.recorder_ is None:
            error_msg = 'The simulation does not record frames. Create it with `record=True`.'
            raise SimulationError(error_msg)
        return self.recorder_.trajectory_

    @staticmethod
    def _partition_problems(problem: BaseProblemSpec) -> tuple[BaseProblemSpec, ...]:
        """Return the problems of the partitions of a compiled problem, none for a problem without partitions."""
        return tuple(partition.problem for partition in problem.partitions)

    @staticmethod
    def _default_dt(until: float) -> float:
        """Return the smallest of 0.01, 0.02, 0.05, 0.1, and so on, that keeps a run until a time within 1000 frames."""
        if not math.isfinite(until):
            return _DEFAULT_DT
        exponent = round(math.log10(_DEFAULT_DT))
        while True:
            for mantissa in _FRAME_MANTISSAS:
                dt = float(f'{mantissa}e{exponent}')
                if until / dt <= _MAX_FRAMES * (1 + TIME_TOLERANCE):
                    return dt
            exponent += 1
