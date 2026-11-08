"""Run compiled systems frame by frame, as fast as possible or paced by a clock."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet
from sklearn.base import clone

from ...base import BaseConfig
from ...components.system import System
from ...errors import ParameterError, SimulationError, ValidationError
from ...specs import AgentStateSpec, EntrySpec, ObservableSpec, RunStateSpec, StepSpec, StructureStateSpec
from ...types import Array, FormKind
from ..backends import BaseBackend
from ..compile import BaseCompiler
from ..compile._observables import ObservableEvaluator
from ..compile._pass_compiler import PassCompiler
from ..compile._rewriting_compiler import RewritingCompiler
from ..compile.passes._operator_compiler import OperatorCompiler
from ..problems import BaseProblemSpec, FrameSpec, ProblemSpec
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
_STEPWISE_FORMS = frozenset(
    {FormKind.DISCRETE, FormKind.AGENT, FormKind.LATTICE, FormKind.MONTE_CARLO, FormKind.LEARNING},
)


class Simulation(BaseConfig[None, float]):
    """Simulation of a system with the solver of its form.

    Each frame advances simulated time by `dt` and is handed to every observer. Parameters change between frames
    only through the schedule, whose values are checked against the domains of their parameters, and `extend`
    continues a run from its last frame.

    A simulation composes the pipeline of the engine and follows its common API: `prepare()` validates the system,
    compiles it and chooses the solver, and `run(until)` prepares and runs from time 0 until a time. A run
    prepares the solver for its end time and every observer on the problem before the first frame, runs every observer
    at every frame, and runs it with `final=True` at the end.

    Args:
        system:
            The system to simulate, or a problem already compiled from one, which skips compiling for repeated runs
            that change only parameters or initial values.

        compiler:
            The compiler of the system, whose backend, solver, and observables the simulation sets, or `None` for the
            compiler of the library's form that the system holds: the operator compiler for a system of operator
            elements, the rewriting compiler for a system of words or patches, and the pass compiler otherwise. A
            system package can bring the compiler of a form of its own, which names the solver of its problems.

        solver:
            The solver, or `None` for the solver the compiler names, or else the default solver of the system's form,
            `CoSimulationSolver` for a system that runs as partitions. The default ODE solver is `ScipyODESolver` with
            `LSODA`, which switches to BDF methods while the system is stiff.

        backend:
            The backend that compiles the system, or `None` for `NumbaBackend` on systems with at least 1000 members and
            `NumpyBackend` on the others.

        dt:
            The simulated time between frames. It defaults to 1 for discrete maps and agent-based systems. Otherwise
            it defaults to 0.01, raised to the smallest of 0.02, 0.05, 0.1, 0.2, and so on, in steps of 1, 2, and 5
            times a power of ten, that keeps a run at most 1000 frames long.

        clock:
            The clock that paces frames against wall time. It defaults to running as fast as possible.

        observers:
            The observers that receive every frame.

        record:
            Whether to record the frames as the trajectory `trajectory_`.

        random_state:
            The seed or generator of stochastic solvers. A seed starts a stream spawned from
            `np.random.SeedSequence(random_state)`, independent of `np.random.default_rng(random_state)`, so a system
            can draw its initial state from the same seed. A generator is used as it is.

        checked:
            Whether to check after every frame that every state lies in its domain, such as an alphabet of labels,
            and raise an error naming the state when it does not.

        schedule:
            A mapping from dotted parameter name to a function of time, such as a ramp of a control parameter. At every
            frame the parameter takes the value of its function at the frame's time and keeps it until the next frame.
            A value outside the domain of its parameter, or one that would break a transition or emission matrix,
            raises an error that names the parameter and the time. An adaptive parameter, such as a bounded control
            input, can be scheduled too, and the trajectory records its scheduled values.

        observables:
            The observables to compute and record at every frame, labelled by their path from the root, such as
            `village/mean_wealth` for an observable of the nested system `village`.

        invariants:
            The tolerance of every observable that must stay constant, by the path of the observable. At every frame
            the run checks that each stays within its tolerance of its value at the first frame, and raises a
            `ConservationError` that names the observable and the first frame that breaks it.

        initial_state:
            Initial values that replace those of the elements, by dotted state name, such as `{'rabbits.n': 20}`.
            The whole state of an operator system is `<system label>.state`, which takes a vector, a density matrix, or
            `'ground'` for the ground state.

        until:
            The simulated time `run` stops at when it is given none, or `None` to require one at every run.

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
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.populations import LogisticGrowthInteraction, PopulationElement
        >>> from skcomplex.core.engine.runtime._simulations import Simulation
        >>> system = System(
        ...     elements=[PopulationElement('rabbits', n=10)],
        ...     interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> simulation = Simulation(system, dt=1.0).prepare().run(until=20)
        >>> round(simulation.trajectory_.values('rabbits.n')[-1])
        100
    """

    def __init__(
        self,
        system: System | BaseProblemSpec,
        *,
        compiler: BaseCompiler | None = None,
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
        until: float | None = None,
    ) -> None:
        self.system = system
        self.compiler = compiler
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
        self.until = until

    def _check_dt(self, problem: BaseProblemSpec, until: float) -> float:
        """Return the time between frames of a run until a time, after validating it for the problem's form."""
        partitions = [item for item in self._partition_problems(problem) if not getattr(item, 'programs', ())]
        forms = {problem.form, *(item.form for item in partitions)}
        stepwise = bool(forms & _STEPWISE_FORMS) or self.stepper_._is_stepwise()
        default = 1.0 if stepwise else self._default_dt(until)
        dt = default if self.dt is None else self.dt
        if not EntrySpec._contains(RealSet.open(0, oo), dt):
            error_msg = f'dt = {dt!r} must be a positive number.'
            raise ValidationError(error_msg)
        if problem.form == FormKind.LEARNING and not EntrySpec._contains(ZZ, dt):
            error_msg = (
                f'dt = {dt!r} is not a whole number. The time between frames of a system that trains is a positive '
                'whole number of epochs.'
            )
            raise ValidationError(error_msg)
        if stepwise and not EntrySpec._contains(ZZ, dt):
            error_msg = f'dt = {dt!r} must be a positive integer for a {problem.form}.'
            raise ValidationError(error_msg)
        return float(dt)

    def _check(self) -> None:
        """Validate the system against the solver, the backend, and the observables, unless it is a compiled problem.

        Raises:
            ValidationError:
                The system is not a system or a compiled problem, such as a system name, or it cannot be compiled or
                run, with every problem found.
        """
        if not isinstance(self.system, System | BaseProblemSpec):
            error_msg = (
                f'A simulation takes a system or a compiled problem, not {self.system!r}. Create a system of the '
                'library from its class, such as `SIRSystem()`.'
            )
            raise ValidationError(error_msg)
        if not isinstance(self.system, BaseProblemSpec):
            self._compiler().prepare(self.system).report_._validated()

    def _prepare(self, context: None) -> None:
        """Compile the system, choose the solver, and gather the clock and the observers of a run.

        Args:
            context:
                Nothing: a simulation prepares on its own system.

        Raises:
            TypeError:
                The compiler names an object that is not a solver.
        """
        initial = dict(self.initial_state or {})
        named: object = None
        if isinstance(self.system, BaseProblemSpec):
            self.problem_ = PassCompiler._with_initial_state(self.system, initial)
        else:
            compiler = self._compiler().prepare(self.system)
            self.problem_ = compiler.run(initial).problem_
            named = compiler.solver_
        if named is not None and not isinstance(named, BaseSolver):
            error_msg = f'A compiler names a solver, not a {type(named).__name__}.'
            raise TypeError(error_msg)
        self._adaptive = self.problem_.adaptive_parameters
        if self.solver is not None:
            self.stepper_ = self.solver
        else:
            self.stepper_ = named if named is not None else self._default_solver(self.problem_.form)
        self.observables_ = ObservableEvaluator(self.problem_).prepare()
        self.clock_ = self.clock if self.clock is not None else AsFastAsPossibleClock()
        self.recorder_ = RecorderObserver() if self.record else None
        monitors = [InvariantObserver(self.invariants)] if self.invariants else []
        self.observers_ = [*monitors, *self.observers, *([self.recorder_] if self.recorder_ is not None else [])]
        self.parameters_ = self.problem_.initial_parameters.copy()

    def _start(self, until: float) -> None:
        """Prepare the solver and the observers for a run from time 0 until a time.

        Args:
            until:
                The simulated time the run stops at.
        """
        self.dt_ = self._check_dt(self.problem_, until)
        self.stepper_.set_params(random_state=self.random_state, until=until).prepare(self.problem_)
        self.time_ = 0.0
        self.state_ = self.stepper_.initial_state_.copy()
        self._step = 0
        self._frame_parameters: npt.NDArray[np.float64] | None = None
        for observer in self.observers_:
            observer.prepare(self.problem_)

    def _get_frame_parameters(self) -> npt.NDArray[np.float64]:
        """Return a copy of the parameters for a frame, shared with the previous frame while they stay the same."""
        if self._frame_parameters is None or not np.array_equal(self._frame_parameters, self.parameters_):
            self._frame_parameters = self.parameters_.copy()
        return self._frame_parameters

    def _get_frame_state(self) -> Array:
        """Return the state of a frame, the current state with its algebraic states solved at the current time."""
        algebraic = self.problem_.algebraic if isinstance(self.problem_, ProblemSpec) else None
        return self.state_ if algebraic is None else algebraic(self.time_, self.state_, self.parameters_)

    def _emit(self, step: int) -> None:
        """Hand the current frame, with the events since the previous one, to every observer."""
        external = self.stepper_.snapshot_
        agents, reporters = (external.groups, external.reporters) if isinstance(external, AgentStateSpec) else ({}, {})
        structures = external.structures if isinstance(external, StructureStateSpec) else {}
        reporters = external.reporters if isinstance(external, StructureStateSpec) else reporters
        state = self._get_frame_state()
        observables = self.observables_.evaluate(self.time_, state, self.parameters_, self.stepper_.agents_.get)
        frame = FrameSpec(
            step,
            self.time_,
            state.copy(),
            self._get_frame_parameters(),
            self.problem_,
            agents,
            reporters,
            observables,
            tuple(self.stepper_.events_),
            structures,
        )
        for observer in self.observers_:
            observer.run(frame)

    def _loop(self, until: float) -> None:
        """Advance frame by frame from the current time until `until`, or until a terminal event.

        The stepper runs once per frame, and its last run, at the last frame or after a stop, releases what it acquired,
        such as an external process.
        """
        origin, first = self.time_, self._step
        self.clock_.prepare()
        self._apply_schedule()
        if first == 0:
            self._emit(0)
        released = False
        try:
            while self.time_ < until - TIME_TOLERANCE:
                self._step += 1
                target = min(self._step * self.dt_, until)
                released = target >= until - TIME_TOLERANCE
                step = StepSpec(self.time_, self.state_, self.parameters_, target)
                self.state_ = self.stepper_.run(step, final=released).state_
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
        """Set every scheduled parameter and adaptive parameter to the value of its function at the current time.

        Raises:
            ParameterError:
                A scheduled name is not a parameter of the problem, or the scheduled values would make a transition or
                emission matrix negative or its rows not sum to 1.

            ValidationError:
                A scheduled value has the wrong shape or is outside the domain of its parameter.
        """
        changes = []
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
            changes.append((name, indices, self._check_value(name, entry, function(self.time_))))
        self._check_matrices(changes)
        for _, indices, value in changes:
            self.parameters_[indices] = value

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
        """Return a scheduled value broadcast to its entry after checking its domain and shape."""
        error_msg = (
            f'At t = {self.time_}, the schedule sets {name} = {value!r}, which is not a value of shape {entry.shape}, '
            f'or of a shape that broadcasts to it, in the domain {entry.domain}. The given value has shape '
            f'{np.shape(value)}.'
        )
        if not entry.contains(value):
            raise ValidationError(error_msg)
        try:
            return entry.encode(value)
        except ValueError as error:
            raise ValidationError(error_msg) from error

    def _check_matrices(self, changes: list[tuple[str, slice, Array]]) -> None:
        """Raise an error when scheduled values would break a transition or emission matrix."""
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
            violation = matrix.run(RunStateSpec(self.time_, self.state_, candidate)).values_ if names else None
            if violation is not None:
                error_msg = (
                    f'At t = {self.time_}, the schedule sets {", ".join(names)}, which would make {violation}. '
                    'Schedule values that keep every entry nonnegative and every row summing to 1.'
                )
                raise ParameterError(error_msg)

    def _check_prepared(self) -> None:
        """Raise an error when the simulation has not been run."""
        if not hasattr(self, 'problem_'):
            error_msg = 'The simulation has not been run. Call `run` first.'
            raise SimulationError(error_msg)

    def _compiler(self) -> BaseCompiler:
        """Return a copy of the compiler with the backend, the solver, and the observables of the simulation."""
        if self.compiler is not None:
            compiler = clone(self.compiler)
        elif isinstance(self.system, System):
            compiler = self._default_compiler(self.system) or PassCompiler()
        else:
            compiler = PassCompiler()
        return compiler.set_params(backend=self.backend, solver=self.solver, observables=self.observables)

    @staticmethod
    def _default_compiler(system: System) -> BaseCompiler | None:
        """Return the compiler of the library's form that a system holds, or `None` for the pass compiler.

        Args:
            system:
                The system.

        Returns:
            compiler:
                The operator compiler for a system of operator elements, the rewriting compiler for a system of words
                or patches, or `None` for any other system.
        """
        if OperatorCompiler._contains_operator(system):
            return OperatorCompiler()
        if RewritingCompiler._is_rewriting(system):
            return RewritingCompiler()
        return None

    def _default_solver(self, form: FormKind | str) -> BaseSolver:
        """Return the default solver of a form, `CoSimulationSolver` for a problem that runs as partitions."""
        if not isinstance(form, FormKind):
            error_msg = f'The {form} form has no default solver. Give a solver, or a compiler that names one.'
            raise ValidationError(error_msg)
        return CoSimulationSolver() if form == FormKind.COUPLED else DEFAULT_SOLVERS[form]()

    def prepare(self, context: None = None) -> Self:
        """Compile the system of the checked simulation and choose the solver.

        Args:
            context:
                Nothing: a simulation prepares on its own system.

        Returns:
            self:
                The prepared simulation, with the compiled problem `problem_` and the solver `stepper_`.
        """
        return super().prepare(context)

    def run(self, until: float | None = None, final: bool = False) -> Self:
        """Run the prepared simulation from time 0 to `until`.

        Args:
            until:
                The simulated time to stop at, or `None` for the `until` of the simulation.

            final:
                Whether this is the last call, after which the simulation releases what preparing acquired.

        Returns:
            simulation:
                The simulation itself.

        Raises:
            NotPreparedError:
                The simulation was not prepared with `prepare`.
            ValidationError:
                Neither the call nor the simulation gives an end time.
        """
        super().run(until, final)
        return self

    def _run(self, inputs: float | None) -> None:
        """Run the prepared simulation from time 0 to `until`.

        The run starts from the initial state and the initial parameter values.

        Args:
            inputs:
                The simulated time to stop at, or `None` for the `until` of the simulation.

        Raises:
            ValidationError:
                Neither the call nor the simulation gives an end time.
        """
        end = self.until if inputs is None else inputs
        if end is None:
            error_msg = 'A simulation runs until a time. Pass it to `run`, such as `run(until=10)`.'
            raise ValidationError(error_msg)
        until = 0.0 if self.problem_.form == FormKind.STATIC else float(end)
        self._start(until)
        self._loop(until)

    def extend(self, until: float) -> Self:
        """Continue a finished run from its last frame to `until`.

        The run keeps its state, parameters, stepper, clock, observers, schedule, and frame duration, and appends its
        frames to the trajectory. A solver that closes an external process at the end of a
        run, such as `LammpsSolver`, cannot continue it.

        Args:
            until:
                The simulated time to stop at, later than the time of the last frame.

        Returns:
            simulation:
                The simulation itself.

        Raises:
            SimulationError:
                The simulation has not run, ended at a terminal event, or `until` is not later than its time.

        Examples:
            >>> from skcomplex.core.components.system import System
            >>> from skcomplex.systems.biology.common.populations import LogisticGrowthInteraction, PopulationElement
            >>> from skcomplex.core.engine.runtime._simulations import Simulation
            >>> system = System(
            ...     elements=[PopulationElement('rabbits', n=10)],
            ...     interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=100)],
            ... )
            >>> schedule = {'growth.r': lambda t: 1.0 if t < 5 else 0.0}
            >>> simulation = Simulation(system, dt=1.0, schedule=schedule).prepare().run(until=5)
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
        if not until > self.time_ + TIME_TOLERANCE:
            error_msg = f'Cannot extend the run to t = {until}, which is not later than its current time {self.time_}.'
            raise SimulationError(error_msg)
        if self.stepper_.stopped_at_ is not None:
            error_msg = f'A terminal event ended the run at t = {self.time_}, so it cannot be extended.'
            raise SimulationError(error_msg)
        self._loop(until)
        return self

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
        return problem.partition_problems

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
