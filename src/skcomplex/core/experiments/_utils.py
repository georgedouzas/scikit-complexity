"""Share the small helpers of the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np
from sklearn.base import clone

from ..components.systems import BaseSystem as CompositeBaseSystem
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec
from ..engine.runtime import Trajectory, simulate
from ..engine.solvers import BaseSolver
from ..errors import ExperimentError, ValidationError
from ..systems import BaseSystem
from .types import AnySystem, SystemSource


def build_system(source: SystemSource, arguments: Mapping[str, object], solver: BaseSolver | None = None) -> AnySystem:
    """Return a copy of the system of a source with parameters set.

    A system is cloned and takes the arguments through `set_params`, as `cross_val_score` clones an estimator, so the
    system of the user never changes. The names are those `set_params` takes, such as the model parameter `r` of a
    ready-made system or the nested name `growth__r` of a `System`. A factory takes them as keyword arguments.

    Args:
        source:
            The system, or the factory of systems.

        arguments:
            The parameters by name.

        solver:
            The solver that replaces the one of a system that holds its solver as a setting, or `None` to keep it.

    Returns:
        system:
            The copy of the system with the parameters.

    Raises:
        ValidationError:
            A name is not a parameter of the system.

        ExperimentError:
            The factory returns no system.
    """
    if isinstance(source, BaseSystem | CompositeBaseSystem):
        deep = isinstance(source, BaseSystem)
        unknown = sorted(set(arguments) - set(source.get_params(deep=deep)))
        if unknown:
            names = ', '.join(sorted(source.get_params(deep=deep)))
            error_msg = (
                f'{", ".join(unknown)} is not a parameter of {type(source).__name__}. Its parameters are {names}, '
                'and dotted paths such as `growth.r` name the states and parameters of its components.'
            )
            raise ValidationError(error_msg)
        system = clone(source).set_params(**arguments)
    else:
        system = source(**arguments)
        if not isinstance(system, BaseSystem | CompositeBaseSystem):
            error_msg = f'The factory of an experiment returns a system, not {type(system).__name__}.'
            raise ExperimentError(error_msg)
    if solver is not None and isinstance(system, BaseSystem):
        system.set_params(solver=solver)
    return system


def get_problem(system: AnySystem) -> BaseProblemSpec:
    """Return the compiled problem of a system.

    Its layouts name the states and parameters of the system.

    A system that is prepared already is not prepared again.

    Args:
        system:
            The system.

    Returns:
        problem:
            The compiled problem.
    """
    if isinstance(system, BaseSystem):
        prepared = system if hasattr(system, 'system_') else system.prepare()
        return prepared.system_._problem
    compiler = system.get_run_settings()['compiler'] or PassCompiler()
    problem: BaseProblemSpec = clone(compiler).prepare(system.prepare().system_).run().problem_
    return problem


def simulate_system(
    system: AnySystem,
    until: float | None,
    *,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | np.random.Generator | None = None,
    replay: Trajectory | None = None,
) -> Trajectory:
    """Return one run of a system.

    A system with `prepare` and `simulate` is prepared the first time it runs, and every later run reuses it. Its solver
    is a setting that `build_system` sets. A system of the old kind runs through the function `simulate`.

    Args:
        system:
            The system.

        until:
            The end time, or `None` for the tuned one.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver of a system of the old kind, or `None` for the tuned one.

        random_state:
            The seed or the generator of the run.

        replay:
            A trajectory whose recorded draws the run replays, or `None`.

    Returns:
        trajectory:
            The trajectory of the run.
    """
    if isinstance(system, BaseSystem):
        prepared = system if hasattr(system, 'system_') else system.prepare()
        return prepared.simulate(until, dt=dt, random_state=random_state, replay=replay)
    return simulate(system, until=until, dt=dt, solver=solver, random_state=random_state, replay=replay)
