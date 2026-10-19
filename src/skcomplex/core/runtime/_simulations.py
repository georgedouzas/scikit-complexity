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
from ..solvers import AgentState, Stepper, resolve_solver
from ._clocks import AsFastAsPossible, Clock
from ._frames import Frame
from ._observers import Observer, Recorder
from ._trajectories import Trajectory

DEFAULT_DT = 0.01
TIME_TOLERANCE = 1e-9


class Simulation(BaseEstimator):
    """Simulation of a system with the solver of its form.

    Each frame advances simulated time by `dt` and is handed to every observer. Parameters can change between
    frames through `update_parameters`, also while a background run started with `start` is going.

    Args:
        system:
            The system to simulate, or a problem already compiled from one, which skips compiling for repeated runs
            that change only parameters or initial values.

        solver:
            `'auto'` for the default solver of the system's form, a registered solver name, or a stepper instance.

        backend:
            The backend that compiles the system, `'auto'`, `'numpy'`, `'numba'`, or a backend instance.

        dt:
            The simulated time between frames. It defaults to 1 for discrete maps and agent-based models and to 0.01
            otherwise.

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
        >>> from skcomplex.models.ecology import LogisticGrowth, Population
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

    def _check_dt(self, problem: Problem) -> float:
        """Return the time between frames after validating it for the problem's form."""
        forms = {problem.form, *(partition.problem.form for partition in problem.partitions)}
        stepwise = bool(forms & {Form.DISCRETE, Form.AGENT, Form.LATTICE, Form.MONTE_CARLO})
        default = 1.0 if stepwise else DEFAULT_DT
        dt = default if self.dt is None else self.dt
        if not Domain.POSITIVE.contains(dt):
            error_msg = f'dt = {dt!r} must be a positive number.'
            raise ValidationError(error_msg)
        if stepwise and not Domain.INTEGER.contains(dt):
            error_msg = f'dt = {dt!r} must be a positive integer for a {problem.form.value}.'
            raise ValidationError(error_msg)
        return float(dt)

    def _prepare(self) -> None:
        """Compile the system, build the stepper, and notify the observers of the start."""
        self.problem_ = self.system if isinstance(self.system, Problem) else compile_system(self.system, self.backend)
        self.dt_ = self._check_dt(self.problem_)
        self.stepper_ = resolve_solver(self.problem_, self.solver)
        self.observables_ = ObservableEvaluator(self.problem_)
        self.stepper_.initialize(self.problem_, np.random.default_rng(self.random_state))
        initial_state = self.stepper_.initial_state(self.problem_.initial_state.copy())
        self.clock_ = self.clock if self.clock is not None else AsFastAsPossible()
        self.recorder_ = Recorder() if self.record else None
        self.observers_ = [*self.observers, *([self.recorder_] if self.recorder_ is not None else [])]
        self.time_ = 0.0
        self.state_ = initial_state
        self.parameters_ = self.problem_.initial_parameters.copy()
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
        for function in (problem.drift, problem.diffusion, problem.propensities, problem.update, problem.boundary):
            if function is not None:
                function(0.0, x, p)
        accelerated = problem.accelerated
        if accelerated is None:
            return
        rng = np.random.default_rng(0)
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
            accelerated.iterate(0.0, x, p, 0.0)

    def _emit(self, step: int) -> None:
        """Hand the current frame, with the events since the previous one, to every observer."""
        external = self.stepper_.snapshot()
        agents, reporters = (external.groups, external.reporters) if isinstance(external, AgentState) else ({}, {})
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
        )
        for observer in self.observers_:
            observer.on_frame(frame)

    def _apply_pending(self) -> None:
        """Apply the parameter changes queued since the last frame."""
        while not self._pending.empty():
            indices, values = self._pending.get_nowait()
            self.parameters_[indices] = values

    def _loop(self, until: float) -> None:
        """Advance frame by frame until `until` or until a stop is requested."""
        step = 0
        self.clock_.start()
        self._apply_schedule()
        self._emit(step)
        try:
            while self.time_ < until - TIME_TOLERANCE and not self._stop_requested.is_set():
                self._running.wait()
                if self._stop_requested.is_set():
                    break
                self._apply_pending()
                step += 1
                target = min(step * self.dt_, until)
                self.state_ = self.stepper_.advance(self.time_, self.state_, self.parameters_, target)
                stopped_at = self.stepper_.stopped_at
                if stopped_at is not None:
                    target = stopped_at
                if self.problem_.boundary is not None:
                    self.state_ = self.problem_.boundary(target, self.state_, self.parameters_)
                self.time_ = target
                self._apply_schedule()
                if self.checked:
                    self._check_domains()
                self._emit(step)
                if stopped_at is not None:
                    break
                self.clock_.wait(self.time_)
        finally:
            for observer in self.observers_:
                observer.on_stop()

    def _apply_schedule(self) -> None:
        """Set every scheduled parameter to the value of its function at the current time."""
        for name, function in (self.schedule or {}).items():
            layout = self.problem_.parameters
            if name not in layout:
                error_msg = f'No parameter named {name!r} to schedule. Available names are: {", ".join(layout.names)}.'
                raise ParameterError(error_msg)
            entry = layout.entry(name)
            self.parameters_[entry.indices] = self._check_value(name, entry, function(self.time_))

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
            return np.broadcast_to(entry.domain.encode(value), entry.shape).ravel().copy()
        except ValueError as error:
            raise ValidationError(error_msg) from error

    def _check_prepared(self) -> None:
        """Raise an error when the simulation has not been run or started."""
        if not hasattr(self, 'problem_'):
            error_msg = 'The simulation has not been run or started. Call `run` or `start` first.'
            raise SimulationError(error_msg)

    def run(self, until: float) -> 'Simulation':
        """Run the simulation from time 0 to `until` in the calling thread.

        Args:
            until:
                The simulated time to stop at.

        Returns:
            simulation:
                The simulation itself.
        """
        self._prepare()
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
        self._prepare()
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
                A mapping from dotted parameter name, such as `growth.r`, to its new value.

        Raises:
            ParameterError:
                A name is not a parameter of the problem.

            ValidationError:
                A value has the wrong shape or is outside the domain of its parameter.
        """
        self._check_prepared()
        layout = self.problem_.parameters
        for name, value in values.items():
            if name not in layout:
                error_msg = f'No parameter named {name!r}. Available names are: {", ".join(layout.names)}.'
                raise ParameterError(error_msg)
            entry = layout.entry(name)
            self._pending.put((entry.indices, self._check_value(name, entry, value)))
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
