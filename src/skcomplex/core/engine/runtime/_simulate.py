"""Define the function that simulates a system and returns its trajectory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from ...base import BaseConfig
from ...components.system import BaseSystem
from ...errors import TagError, ValidationError
from ...specs import ObservableSpec, Tags
from ..backends import BaseBackend
from ..compile import BaseCompiler
from ..solvers import BaseSolver
from ..solvers._defaults import DEFAULT_SOLVERS
from ._base import BaseObserver
from ._simulations import Simulation
from ._trajectories import Trajectory


def _check_support(runner: BaseConfig, kind: str, tags: Tags) -> None:
    """Raise an error for a solver or a backend that does not support the tags of a system.

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
    missing = supported.missing(tags)
    if missing:
        name, values = next(iter(missing.items()))
        error_msg = (
            f'The {kind} {type(runner).__name__} does not support the {name} {", ".join(values)} of the system. Pass '
            f'a {kind} that supports them, or none for the default.'
        )
        raise TagError(error_msg)


def _clone_setting[ConfigT: BaseConfig](setting: ConfigT | None) -> ConfigT | None:
    """Return an unfitted copy of a solver, a backend, or a compiler, so a run never changes the one it was given.

    Args:
        setting:
            The solver, the backend, or the compiler, or `None`.

    Returns:
        copy:
            The copy, or `None`.
    """
    return None if setting is None else clone(setting)


def _choose_solver(tags: Tags) -> BaseSolver | None:
    """Return the default solver the tags choose, or `None` for the compiled form of the system to choose it.

    The tags choose the default solver when it is the only default solver of a form that supports them. When law
    kinds are unknown, or the tags fit the default solvers of several forms, such as an update law that a map, a
    lattice, or an agent-based solver runs, the form the compiler infers from the law classes chooses.

    Args:
        tags:
            The tags of the system.

    Returns:
        solver:
            The default solver, or `None`.
    """
    if tags.laws is None:
        return None
    fitting = {solver for solver in DEFAULT_SOLVERS.values() if not solver.get_supported_tags().missing(tags)}
    return fitting.pop()() if len(fitting) == 1 else None


def simulate(
    system: BaseSystem,
    *,
    until: float | None = None,
    dt: float | None = None,
    initial_state: Mapping[str, object] | None = None,
    solver: BaseSolver | None = None,
    backend: BaseBackend | None = None,
    observables: Sequence[ObservableSpec] = (),
    observers: Sequence[BaseObserver] = (),
    invariants: Mapping[str, float] | None = None,
    schedule: Mapping[str, Callable[[float], float | npt.ArrayLike]] | None = None,
    compiler: BaseCompiler | None = None,
    random_state: int | np.random.Generator | None = None,
) -> Trajectory:
    """Simulate a system until a time and return its trajectory.

    A solver takes the role of a splitter in scikit-learn, a backend the role of `n_jobs`, observables the role of
    `scoring`, and observers, such as a Rerun recording, the role of callbacks. An argument left out comes from the
    tuned run settings of the system, which `get_run_settings` returns. A system brings a form of its own through its
    tuned `compiler`, the one hook to the compiled form.
    Without a solver, the tags of the system choose the default solver, or the compiled form chooses it when the tags
    fit several, when the system brings its own compiler, or when it nests systems that run as partitions. The tags
    are read before any compiling, so a solver or a backend that does not support them is rejected first.

    Args:
        system:
            The system to simulate.

        until:
            The end time of the run, or `None` for the tuned end time of the class of the system.

        dt:
            The time between frames, or `None` for the tuned time of the class of the system, or else the default of
            its form.

        initial_state:
            Initial values that replace those of the elements, by dotted state name, such as `{'rabbits.n': 20}`, or
            `None` for the tuned initial state of the class of the system.

        solver:
            The solver, or `None` for the tuned solver of the class of the system, or else the default solver.

        backend:
            The backend, or `None` for the tuned backend of the system, or else the default backend.

        observables:
            The observables to compute and record at every frame, or none for the tuned observables of the class of
            the system.

        observers:
            The observers that receive every frame, such as a `RerunObserver`.

        invariants:
            The tolerance of every observable that must stay constant, by the path of the observable, or `None` for the
            tuned invariants of the system. The run raises a `ConservationError` that names the first observable and
            frame that leave the tolerance of their value at the first frame.

        schedule:
            A mapping from dotted parameter name to a function of time, such as a ramp of a control parameter, or
            `None` for the tuned schedule of the system. At every frame the parameter takes the value of its function
            at the frame's time.

        compiler:
            The compiler of the system, or `None` for the tuned compiler of the system, or else the compiler of the
            library's form the system holds, such as the rewriting form of words and patches or the operator form of
            quantum elements. A system package that brings a form of its own tunes a compiler it defines on
            `BaseCompiler`, the one documented hook to the compiled form.

        random_state:
            The seed or generator of stochastic solvers.

    Returns:
        trajectory:
            The trajectory of the system from time 0 until the end time.

    Raises:
        ValidationError:
            The system is not a system, or no end time is given or tuned.

        TagError:
            The parts of the system disagree on a tag, or the solver or the backend does not support a tag of the
            system, named in the message.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.systems.biology.common.populations import LogisticGrowthInteraction, PopulationElement
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = System(
        ...     elements=[PopulationElement('rabbits', n=10)],
        ...     interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> round(simulate(system, until=20).values('rabbits.n')[-1])
        100
    """
    if not isinstance(system, BaseSystem):
        error_msg = f'simulate takes a system, not {system!r}.'
        raise ValidationError(error_msg)
    settings = system.get_run_settings()
    end = settings['until'] if until is None else until
    if end is None:
        error_msg = f'{type(system).__name__} has no tuned end time. Pass `until`.'
        raise ValidationError(error_msg)
    stepper = _clone_setting(settings['solver'] if solver is None else solver)
    runner = _clone_setting(settings['backend'] if backend is None else backend)
    form = _clone_setting(settings['compiler'] if compiler is None else compiler)
    tags = system.get_tags()
    if stepper is not None:
        _check_support(stepper, 'solver', tags)
    if runner is not None:
        _check_support(runner, 'backend', tags)
    composite = system.prepare().system_
    form = Simulation._default_compiler(composite) if form is None else form
    tuned_observables = () if composite is system else settings['observables']
    if stepper is None and form is None and not composite.systems:
        stepper = _choose_solver(tags)
    simulation = Simulation(
        composite,
        compiler=form,
        solver=stepper,
        backend=runner,
        dt=settings['dt'] if dt is None else dt,
        observers=observers,
        schedule=settings['schedule'] if schedule is None else schedule,
        observables=observables or tuned_observables,
        invariants=settings['invariants'] if invariants is None else invariants,
        initial_state=settings['initial_state'] if initial_state is None else initial_state,
        random_state=random_state,
        until=end,
    )
    trajectory: Trajectory = simulation.prepare().run().trajectory_
    return trajectory
