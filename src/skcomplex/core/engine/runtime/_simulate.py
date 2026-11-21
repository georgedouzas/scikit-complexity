"""Define the functions that compile a system and run it into a trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from ...base import BaseConfig
from ...components import Observable, Tags
from ...errors import TagError, ValidationError
from ..backends import BaseBackend, NumpyBackend
from ..compile import BaseCompiler
from ..compile._pass_compiler import PassCompiler
from ..problems import BaseProblemSpec
from ..problems._composite_system import _CompositeSystem
from ..solvers import BaseSolver
from ..solvers._defaults import DEFAULT_SOLVERS
from ._base import BaseClock, BaseObserver
from ._simulations import Simulation
from ._trajectories import Trajectory


def _check_support(runner: BaseConfig, kind: str, tags: Tags) -> None:
    """Raise an error for a solver or a backend that does not support a system's tags.

    Args:
        runner:
            The solver or the backend.

        kind:
            `'solver'` or `'backend'`.

        tags:
            The tags of the system.

    Raises:
        TagError:
            The solver or the backend does not support a tag, named in the message.
    """
    get_supported = getattr(type(runner), 'get_supported_tags', None)
    if not callable(get_supported):
        error_msg = f'The {kind} {runner!r} is not a {kind} of the engine.'
        raise ValidationError(error_msg)
    supported: Tags = get_supported()
    missing = supported.get_missing(tags)
    if missing:
        name, values = next(iter(missing.items()))
        error_msg = (
            f'The {kind} {type(runner).__name__} does not support the {name} {", ".join(values)} of the system. Pass '
            f'a {kind} that supports them, or none for the default.'
        )
        raise TagError(error_msg)


def _clone_setting[ConfigT: BaseConfig](setting: ConfigT | None) -> ConfigT | None:
    """Return an unfitted copy of a solver, a backend, or a compiler.

    A run thus never changes the one it was given.

    Args:
        setting:
            The solver, the backend, or the compiler, or `None`.

    Returns:
        copy:
            The copy, or `None`.
    """
    return None if setting is None else clone(setting)


def _choose_solver(tags: Tags) -> BaseSolver | None:
    """Return the default solver the tags choose.

    It returns `None` for the compiled form of the system to choose it. The tags choose the default solver when it is
    the only default solver of a form that supports them. When law kinds are unknown, or the tags fit the default
    solvers of several forms, such as an update law that a map, a lattice, or an agent-based solver runs, the form the
    compiler infers from the law classes chooses. The compiled problem also chooses for noise laws, whose tags do not
    tell fractional noise, which `FractionalEulerMaruyamaSolver` runs, from ordinary noise.

    Args:
        tags:
            The tags of the system.

    Returns:
        solver:
            The default solver, or `None`.
    """
    if tags.laws is None or 'noise' in tags.laws:
        return None
    fitting = {solver for solver in DEFAULT_SOLVERS.values() if not solver.get_supported_tags().get_missing(tags)}
    return fitting.pop()() if len(fitting) == 1 else None


def _prepare_runners(
    system: _CompositeSystem,
    solver: BaseSolver | None,
    backend: BaseBackend | None,
    observables: Sequence[Observable],
) -> tuple[_CompositeSystem, BaseSolver | None, BaseBackend | None, BaseCompiler | None, Sequence[Observable]]:
    """Return the composite of a system with copies of its runners, and observables.

    The runners are the solver, the backend, and the compiler the system needs. A setting left out comes from the tuned
    run settings of the system. The tags are read before any compiling, so a solver or a backend that does not support
    them is rejected first, and without a solver the tags choose the default one, unless the compiled form chooses it.

    Args:
        system:
            The system.

        solver:
            The solver, or `None` for the tuned or the default one.

        backend:
            The backend, or `None` for the tuned one.

        observables:
            The observables, or none for the tuned ones.

    Returns:
        composite:
            The composite of the components of the system.

        solver:
            The copy of the solver, or `None` for the compiled form to choose it.

        backend:
            The copy of the backend, or `None`.

        compiler:
            The copy of the compiler, or `None` for the pass compiler.

        observables:
            The observables of the run.

    Raises:
        TagError:
            The components of the system disagree on a tag, or the solver or the backend does not support a tag.
    """
    settings = system.get_run_settings()
    stepper = _clone_setting(settings['solver'] if solver is None else solver)
    runner = _clone_setting(settings['backend'] if backend is None else backend)
    declared = PassCompiler()._get_declared_system(system) if isinstance(system, _CompositeSystem) else system
    tags = declared.get_tags()
    if stepper is not None:
        _check_support(stepper, 'solver', tags)
    if runner is not None:
        _check_support(runner, 'backend', tags)
    composite = system.prepare().system_
    form = Simulation._get_default_compiler(composite)
    tuned_observables = () if isinstance(system, _CompositeSystem) else settings['observables']
    if stepper is None and form is None and not composite.systems:
        stepper = _choose_solver(tags)
    return composite, stepper, runner, form, observables or tuned_observables


def compile_problem(
    system: _CompositeSystem,
    *,
    solver: BaseSolver | None = None,
    backend: BaseBackend | None = None,
    observables: Sequence[Observable] = (),
) -> tuple[BaseProblemSpec, BaseSolver]:
    """Return the problem compiled from a system, and the solver of its runs.

    It turns the laws of a system into a problem once, such as the problem of one state `'population.x'` of a logistic
    growth, so many runs through `run_problem` that change only starting values, schedules, or seeds skip compiling. An
    experiment compiles a system once per structure this way, as for an ensemble of epidemics, a sweep of the rates of
    a reaction, or a fit of a market model.

    Read more in the user guide, under Run experiments.

    Args:
        system:
            The system.

        solver:
            The solver, or `None` for the tuned or the default one.

        backend:
            The backend the problem is compiled for, such as `NumbaBackend()`, or `None` for the tuned or the default
            one.

        observables:
            The observables, or none for the tuned ones.

    Returns:
        problem:
            The compiled problem, with the initial values of the system.

        solver:
            The solver of the runs, which every run copies.

    See Also:
        run_problem: Runs a system from the problem it returns.

    Examples:
        >>> from skcomplex.core.engine.runtime import compile_problem
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> problem, solver = compile_problem(LogisticGrowthSystem()._get_composite())
        >>> problem.states.names, type(solver).__name__
        (['population.x'], 'ScipyODESolver')
    """
    composite, stepper, runner, form, observed = _prepare_runners(system, solver, backend, observables)
    simulation = Simulation(composite, compiler=form, solver=stepper, backend=runner, observables=observed).prepare()
    return simulation.problem_, simulation.stepper_


def run_problem(
    system: _CompositeSystem,
    compiled: tuple[BaseProblemSpec, BaseSolver] | None,
    *,
    until: float | None = None,
    dt: float | None = None,
    initial_state: Mapping[str, object] | None = None,
    solver: BaseSolver | None = None,
    backend: BaseBackend | None = None,
    observables: Sequence[Observable] = (),
    observers: Sequence[BaseObserver] = (),
    invariants: Mapping[str, float] | None = None,
    schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None = None,
    random_state: int | np.random.Generator | None = None,
    checked: bool = False,
    clock: BaseClock | None = None,
    record_draws: bool = False,
    replay: Trajectory | None = None,
) -> Trajectory:
    """Return the trajectory of a run of a system, as `simulate` does.

    It runs a system from time 0 until `until` and starts from a problem that `compile_problem` built when one is given,
    so runs that change only starting values, such as `{'population.x': 2.0}`, skip compiling. `System.simulate` and the
    experiments call it, for the runs of an epidemic, a reaction, or a market alike.

    Read more in the user guide, under Run experiments.

    Args:
        system:
            The system to simulate.

        compiled:
            The problem compiled from the system with the solver of its runs, or `None` to compile the system.

        until:
            The end time of the run, or `None` for the tuned one.

        dt:
            The time between frames, or `None` for the tuned one.

        initial_state:
            Initial values by dotted state name, or `None` for the tuned initial state.

        solver:
            The solver, or `None` for the tuned or the default one, or the one of the compiled problem.

        backend:
            The backend, or `None` for the tuned one.

        observables:
            The observables, or none for the tuned ones.

        observers:
            The observers that receive every frame.

        invariants:
            The tolerance of every observable that must stay constant, or `None` for the tuned invariants.

        schedule:
            A function of time per dotted parameter name, or `None` for the tuned schedule.


        random_state:
            The seed or generator of stochastic solvers.

        checked:
            Whether to check after every frame that every state lies in its domain.

        clock:
            The clock that paces frames against wall time, or `None` to run as fast as possible.

        record_draws:
            Whether to record the named draws and jump firings of the run.

        replay:
            A trajectory recorded with its draws, whose draws the run forces, or `None`.

    Returns:
        trajectory:
            The trajectory of the system from time 0 until the end time.

    Raises:
        ValidationError:
            The system is not a system, or no end time is given or tuned.

    See Also:
        compile_problem: Compiles the problem it starts from.

    Examples:
        >>> from skcomplex.core.engine.runtime import run_problem
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> composite = LogisticGrowthSystem()._get_composite()
        >>> run_problem(composite, None, until=2, dt=1, initial_state={'population.x': 2.0}).times.tolist()
        [0.0, 1.0, 2.0]
    """
    if not isinstance(system, _CompositeSystem):
        error_msg = f'simulate takes a system, not {system!r}.'
        raise ValidationError(error_msg)
    settings = system.get_run_settings()
    end = settings['until'] if until is None else until
    if end is None:
        error_msg = f'{type(system).__name__} has no tuned end time. Pass `until`.'
        raise ValidationError(error_msg)
    if (record_draws or replay is not None) and backend is None:
        backend = NumpyBackend()
    source: _CompositeSystem | BaseProblemSpec
    if compiled is None:
        source, stepper, runner, form, observed = _prepare_runners(system, solver, backend, observables)
    else:
        source, runner, form, observed = compiled[0], _clone_setting(backend), None, ()
        stepper = clone(compiled[1] if solver is None else solver)
    simulation = Simulation(
        source,
        compiler=form,
        solver=stepper,
        backend=runner,
        dt=settings['dt'] if dt is None else dt,
        observers=observers,
        schedule=settings['schedule'] if schedule is None else schedule,
        observables=observed,
        invariants=settings['invariants'] if invariants is None else invariants,
        initial_state=settings['initial_state'] if initial_state is None else initial_state,
        random_state=random_state,
        checked=checked,
        clock=clock,
        until=end,
        record_draws=record_draws,
        replay=replay,
    )
    trajectory: Trajectory = simulation.prepare().run().trajectory_
    return trajectory
