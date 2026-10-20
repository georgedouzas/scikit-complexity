"""Run compiled systems frame by frame, offline, in real time, or in the background."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
import queue
import threading
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from ..base import Domain, Form, ParameterError, SimulationError, ValidationError
from ..compile import Backend, Entry, ObservableEvaluator, Problem, Vector, compile_system
from ..model import System
from ..solvers import AgentState, RewritingState, Stepper, resolve_solver
from ._clocks import AsFastAsPossible, Clock
from ._frames import Frame
from ._observers import Observer, Recorder
from ._trajectories import Trajectory

DEFAULT_DT = 0.01
MAX_FRAMES = 1000
FRAME_MANTISSAS = (1, 2, 5)
TIME_TOLERANCE = 1e-9


def _default_dt(until: float) -> float:
    """Return the smallest of 0.01, 0.02, 0.05, 0.1, and so on, that keeps a run until a time at most 1000 frames."""
    if not math.isfinite(until):
        return DEFAULT_DT
    exponent = round(math.log10(DEFAULT_DT))
    while True:
        for mantissa in FRAME_MANTISSAS:
            dt = float(f'{mantissa}e{exponent}')
            if until / dt <= MAX_FRAMES * (1 + TIME_TOLERANCE):
                return dt
        exponent += 1


class Simulation(BaseEstimator):
    """Simulation of a system with the solver of its form.

    Each frame advances simulated time by `dt` and is handed to every observer. Parameters can change between
    frames through `update_parameters`, also while a background run started with `start` is going. Values set
    between runs stay in force for the later runs of the simulation, and `extend` continues a run from its last
    frame.

    Args:
        system:
            The system to simulate, or a problem already compiled from one, which skips compiling for repeated runs
            that change only parameters or initial values.

        solver:
            `'auto'` for the default solver of the system's form, a registered solver name, or a stepper instance.
            The default ODE solver is `ScipyODE` with `LSODA`, which switches to BDF methods while the system is
            stiff.

        backend:
            The backend that compiles the system, `'auto'`, `'numpy'`, `'numba'`, or a backend instance.

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
        >>> from skcomplex.core.model import System
        >>> from skcomplex.models.biology.ecology import LogisticGrowth, Population
        >>> from skcomplex.core.runtime import Simulation
        >>> system = System(
        ...     elements=[Population('rabbits', n=10)],
        ...     interactions=[LogisticGrowth('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> simulation = Simulation(system, dt=1.0).run(until=20)
        >>> round(simulation.trajectory_['rabbits.n'][-1])
        100
    """

    def __init__(
        self,
        system: System | Problem,
        *,
        solver: str | Stepper = 'auto',
        backend: str | Backend = 'auto',
        dt: float | None = None,
        clock: Clock | None = None,
        observers: Sequence[Observer] = (),
        record: bool = True,
        random_state: int | np.random.Generator | None = None,
        checked: bool = False,
        schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None = None,
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

    def _check_dt(self, problem: Problem, until: float) -> float:
        """Return the time between frames of a run until a time, after validating it for the problem's form."""
        forms = {problem.form, *(partition.problem.form for partition in problem.partitions)}
        stepwise = bool(forms & {Form.DISCRETE, Form.AGENT, Form.LATTICE, Form.MONTE_CARLO, Form.REWRITING})
        default = 1.0 if stepwise else _default_dt(until)
        dt = default if self.dt is None else self.dt
        if not Domain.POSITIVE.contains(dt):
            error_msg = f'dt = {dt!r} must be a positive number.'
            raise ValidationError(error_msg)
        if stepwise and not Domain.INTEGER.contains(dt):
            error_msg = f'dt = {dt!r} must be a positive integer for a {problem.form.value}.'
            raise ValidationError(error_msg)
        return float(dt)

    def _previous_parameters(self) -> tuple[list[str], Vector] | None:
        """Return the parameter names and values of the previous run of the same system, or `None` without one."""
        if getattr(self, '_prepared_system', None) is not self.system or not hasattr(self, 'parameters_'):
            return None
        problem: Problem = self.problem_
        return problem.parameters.names, self.parameters_.copy()

    def _prepare(self, until: float) -> None:
        """Compile the system, build the stepper, and notify the observers of the start."""
        previous = self._previous_parameters()
        self.problem_ = self.system if isinstance(self.system, Problem) else compile_system(self.system, self.backend)
        self._prepared_system = self.system
        self.dt_ = self._check_dt(self.problem_, until)
        self.stepper_ = resolve_solver(self.problem_, self.solver)
        self.observables_ = ObservableEvaluator(self.problem_)
        self.stepper_.initialize(self.problem_, np.random.default_rng(self.random_state))
        self.stepper_.plan(until)
        initial_state = self.stepper_.initial_state(self.problem_.initial_state.copy())
        self.clock_ = self.clock if self.clock is not None else AsFastAsPossible()
        self.recorder_ = Recorder() if self.record else None
        self.observers_ = [*self.observers, *([self.recorder_] if self.recorder_ is not None else [])]
        self.time_ = 0.0
        self.state_ = initial_state
        self.parameters_ = self.problem_.initial_parameters.copy()
        if previous is not None and previous[0] == self.problem_.parameters.names:
            self.parameters_ = previous[1] if previous[1].shape == self.parameters_.shape else self.parameters_
        self._step = 0
        self._pending: queue.Queue[tuple[slice, Vector]] = queue.Queue()
        self._stop_requested = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._compile_functions()
        for observer in self.observers_:
            observer.on_start(self.problem_)

    def _compile_functions(self) -> None:
        """Call every numeric function of the problem and its partitions once, before the clock starts."""
        for problem in (self.problem_, *(partition.problem for partition in self.problem_.partitions)):
            self._compile_problem(problem)

    def _compile_problem(self, problem: Problem) -> None:
        """Call every numeric function of one problem once, so compiled backends finish compiling."""
        x, p = self.state_, self.parameters_
        rng = np.random.default_rng(0)
        drawn = (rng,) if problem.stochastic else ()
        for function in (problem.drift, problem.diffusion, problem.propensities, problem.boundary):
            if function is not None:
                function(0.0, x, p)
        if problem.update is not None:
            problem.update(0.0, x, p, *drawn)
        accelerated = problem.accelerated
        if accelerated is None:
            return
        if accelerated.rk4 is not None:
            accelerated.rk4(0.0, x, p, 0.0, 1.0)
        if accelerated.velocity_verlet is not None:
            empty = np.zeros(0, dtype=np.int64)
            accelerated.velocity_verlet(0.0, x, p, 0.0, 1.0, empty, empty)
        if accelerated.euler_maruyama is not None:
            accelerated.euler_maruyama(0.0, x, p, 0.0, 1.0, rng)
        if accelerated.gillespie is not None:
            accelerated.gillespie(0.0, x, p, 0.0, rng)
        if accelerated.iterate is not None:
            accelerated.iterate(0.0, x, p, 0.0, *drawn)

    def _emit(self, step: int) -> None:
        """Hand the current frame, with the events since the previous one, to every observer."""
        external = self.stepper_.snapshot()
        agents, reporters = (external.groups, external.reporters) if isinstance(external, AgentState) else ({}, {})
        structures = external.structures if isinstance(external, RewritingState) else {}
        reporters = external.reporters if isinstance(external, RewritingState) else reporters
        observables = self.observables_.evaluate(self.time_, self.state_, self.parameters_, self.stepper_.agents_of)
        frame = Frame(
            step,
            self.time_,
            self.state_.copy(),
            self.parameters_.copy(),
            self.problem_,
            agents,
            reporters,
            observables,
            tuple(self.stepper_.drain_events()),
            structures,
        )
        for observer in self.observers_:
            observer.on_frame(frame)

    def _apply_pending(self) -> None:
        """Apply the parameter changes queued since the last frame."""
        while not self._pending.empty():
            indices, values = self._pending.get_nowait()
            self.parameters_[indices] = values

    def _loop(self, until: float) -> None:
        """Advance frame by frame from the current time until `until` or until a stop is requested."""
        origin, first = self.time_, self._step
        self.clock_.start()
        self._apply_schedule()
        if first == 0:
            self._emit(0)
        try:
            while self.time_ < until - TIME_TOLERANCE and not self._stop_requested.is_set():
                self._running.wait()
                if self._stop_requested.is_set():
                    break
                self._apply_pending()
                self._step += 1
                target = min(self._step * self.dt_, until)
                self.state_ = self.stepper_.advance(self.time_, self.state_, self.parameters_, target)
                stopped_at = self.stepper_.stopped_at
                if stopped_at is not None:
                    target = stopped_at
                self.time_ = target
                self._apply_schedule()
                if self.checked:
                    self._check_domains()
                self._emit(self._step)
                if stopped_at is not None:
                    break
                self.clock_.wait(self.time_ - origin)
        finally:
            for observer in self.observers_:
                observer.on_stop()

    def _apply_schedule(self) -> None:
        """Set every scheduled parameter to the value of its function at the current time."""
        for name, function in (self.schedule or {}).items():
            layout = self.problem_.parameters
            try:
                entry, indices = layout.locate(name)
            except KeyError as error:
                error_msg = f'No parameter named {name!r} to schedule. Available names are: {", ".join(layout.names)}.'
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
        for entry in self.problem_.states:
            values = np.asarray(self.problem_.states.read(self.state_, entry.name))
            if not entry.domain.contains(values):
                error_msg = (
                    f'{entry.name} took the values {np.round(values, 6).tolist()} at t = {self.time_}, outside its '
                    f'{entry.domain.value} domain. Check the laws that change it.'
                )
                raise ValidationError(error_msg)

    def _check_value(self, name: str, entry: Entry, value: float | npt.ArrayLike) -> Vector:
        """Return a parameter value broadcast to its entry after checking its domain and shape."""
        error_msg = (
            f'{name} = {value!r} is not a value of shape {entry.shape}, or of a shape that broadcasts to it, in the '
            f'{entry.domain.value} domain.'
        )
        if not entry.domain.contains(value):
            raise ValidationError(error_msg)
        try:
            return entry.encode(value)
        except ValueError as error:
            raise ValidationError(error_msg) from error

    def _check_matrices(self, changes: list[tuple[str, slice, Vector]]) -> None:
        """Raise an error when parameter changes would break a transition or emission matrix."""
        problems = (self.problem_, *(partition.problem for partition in self.problem_.partitions))
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
            violation = matrix.violation(candidate) if names else None
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

    def run(self, until: float) -> 'Simulation':
        """Run the simulation from time 0 to `until` in the calling thread.

        The run starts from the initial state and keeps the parameter values of the previous run of the same system,
        such as values set with `update_parameters` between runs. `clone(simulation).run(until)` starts afresh.

        Args:
            until:
                The simulated time to stop at.

        Returns:
            simulation:
                The simulation itself.
        """
        self._prepare(until)
        self._loop(until)
        return self

    def extend(self, until: float) -> 'Simulation':
        """Continue a finished run from its last frame to `until` in the calling thread.

        The run keeps its state, parameters, stepper, clock, observers, and frame duration, applies pending parameter
        changes, and appends its frames to the trajectory.

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
            >>> from skcomplex.core.model import System
            >>> from skcomplex.models.biology.ecology import LogisticGrowth, Population
            >>> from skcomplex.core.runtime import Simulation
            >>> system = System(
            ...     elements=[Population('rabbits', n=10)],
            ...     interactions=[LogisticGrowth('growth', target='rabbits', r=1.0, K=100)],
            ... )
            >>> simulation = Simulation(system, dt=1.0).run(until=5)
            >>> simulation.update_parameters({'growth.r': 0.0})
            >>> trajectory = simulation.extend(until=10).trajectory_
            >>> trajectory.times.tolist()
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            >>> bool(trajectory['rabbits.n'][5] == trajectory['rabbits.n'][10])
            True
        """
        self._check_prepared()
        if self._thread is not None and self._thread.is_alive():
            error_msg = 'The simulation is running in the background. Call `wait` or `stop` before `extend`.'
            raise SimulationError(error_msg)
        if not until > self.time_ + TIME_TOLERANCE:
            error_msg = f'Cannot extend the run to t = {until}, which is not later than its current time {self.time_}.'
            raise SimulationError(error_msg)
        if self.stepper_.stopped_at is not None:
            error_msg = f'A terminal event ended the run at t = {self.time_}, so it cannot be extended.'
            raise SimulationError(error_msg)
        self._stop_requested.clear()
        self._running.set()
        self._loop(until)
        return self

    def start(self, until: float = math.inf) -> 'Simulation':
        """Start the simulation from time 0 in a background thread.

        Args:
            until:
                The simulated time to stop at. The default runs until `stop` is called.

        Returns:
            simulation:
                The simulation itself.
        """
        self._prepare(until)
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
        return self.recorder_.trajectory
