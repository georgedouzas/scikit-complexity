"""Share the small helpers of the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np
from sklearn.base import clone

from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec
from ..engine.problems._composite_system import _CompositeSystem
from ..engine.runtime import Trajectory, run_problem
from ..engine.solvers import BaseSolver
from ..errors import ExperimentError, ValidationError
from ..systems import BaseSystem
from .types import AnySystem, SystemSource


def build_system(source: SystemSource, arguments: Mapping[str, object], solver: BaseSolver | None = None) -> AnySystem:
    """Return a copy of the system of a source with parameters set.

    It clones a system and sets the parameters with `set_params`, as `cross_val_score` clones an estimator, so the
    system of the user never changes, or it calls a factory with them, such as `{'r': 0.3}` for the growth rate of a
    population. Every experiment builds its runs with it, for an epidemic, a market, or a magnet alike.

    Read more in the user guide, under Run experiments.

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

    See Also:
        simulate_system: Runs the system it returns.

    Examples:
        >>> from skcomplex.core.experiments._utils import build_system
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> build_system(LogisticGrowthSystem(), {'r': 0.3}).r
        0.3
    """
    if isinstance(source, BaseSystem | _CompositeSystem):
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
        if not isinstance(system, BaseSystem | _CompositeSystem):
            error_msg = f'The factory of an experiment returns a system, not {type(system).__name__}.'
            raise ExperimentError(error_msg)
    if solver is not None and isinstance(system, BaseSystem):
        system.set_params(solver=solver)
    return system


def get_problem(system: AnySystem) -> BaseProblemSpec:
    """Return the compiled problem of a system.

    It prepares the system when it is not prepared yet and returns the problem its laws compile into, whose layouts name
    the states and the parameters, such as the growth rate of a population. Experiments read it to find the values they
    set, for an epidemic, a reaction, or a market alike.

    Read more in the user guide, under Run experiments.

    Args:
        system:
            The system.

    Returns:
        problem:
            The compiled problem.

    See Also:
        build_system: Returns the copy of a system with parameters set.

    Examples:
        >>> from skcomplex.core.experiments._utils import get_problem
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> get_problem(LogisticGrowthSystem()).backend
        'numpy'
    """
    if isinstance(system, BaseSystem):
        prepared = system if hasattr(system, 'system_') else system.prepare()
        return prepared.system_._problem
    problem: BaseProblemSpec = PassCompiler().prepare(system.prepare().system_).run().problem_
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

    It prepares the system the first time it runs and reuses it later, then simulates it to `until`, such as five units
    of time of a growing population. Every experiment runs its systems through it, for an epidemic, a market, or a
    magnet alike.

    Read more in the user guide, under Run experiments.

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

    See Also:
        build_system: Returns the copy of a system with parameters set.

    Examples:
        >>> from skcomplex.core.experiments._utils import simulate_system
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> float(simulate_system(LogisticGrowthSystem(), until=5.0).times[-1])
        5.0
    """
    if isinstance(system, BaseSystem):
        prepared = system if hasattr(system, 'system_') else system.prepare()
        return prepared.simulate(until, dt=dt, random_state=random_state, replay=replay)
    return run_problem(system, None, until=until, dt=dt, solver=solver, random_state=random_state, replay=replay)
