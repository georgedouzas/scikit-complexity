"""Share the small helpers of the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import cache
from importlib import import_module
from types import ModuleType

from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec, ProblemSpec
from ..engine.problems._composite_system import _CompositeSystem
from ..errors import AnalysisError


@cache
def load_module(name: str) -> ModuleType:
    """Return a module, importing it on the first call.

    The analyses import heavy optional libraries, such as `dit` or `pysindy`, only when they first need them, and
    later calls return the same module at once, such as the `math` module for `load_module('math')`. It keeps the
    import of the package fast for studies of epidemics, markets, or flows that never touch those libraries.

    Read more in the user guide, under Analyse.

    Args:
        name:
            The dotted name of the module.

    Returns:
        module:
            The imported module.

    See Also:
        compile_problem: Returns the problem of the general compiler of a system.

    Examples:
        >>> from skcomplex.core.analysis._utils import load_module
        >>> load_module('math').sqrt(4.0)
        2.0
    """
    return import_module(name)


def compile_problem(system: _CompositeSystem | BaseProblemSpec, purpose: str) -> ProblemSpec:
    """Return the problem of the general compiler of a system, or the problem itself.

    The analyses that read the states, the parameters, and the laws of a system, such as `stability` or `loops`, call
    it to get the compiled problem, such as the problem of ordinary differential equations of a spring. It serves the
    analyses of an epidemic, a circuit, or an economy alike.

    Read more in the user guide, under Analyse.

    Args:
        system:
            A system, or a problem compiled from one.

        purpose:
            What needs the general compiler's problem, such as `'Linear stability'`, for the error message.

    Returns:
        problem:
            The problem of the general compiler.

    Raises:
        AnalysisError:
            A compiler of whole systems compiles the system to a problem of its own, which the work cannot read.

    See Also:
        load_module: Imports a module on the first call.

    Examples:
        >>> from skcomplex.core.analysis._utils import compile_problem
        >>> from skcomplex.systems.physics.mechanics import HarmonicOscillatorSystem
        >>> trajectory = HarmonicOscillatorSystem().prepare().simulate(until=1.0)
        >>> compile_problem(trajectory.problem, 'Linear stability').states.names
        ['mass.x', 'mass.x_d1']
    """
    problem = system if isinstance(system, BaseProblemSpec) else PassCompiler().prepare(system).run().problem_
    if isinstance(problem, ProblemSpec):
        return problem
    error_msg = (
        f'{purpose} reads the problem of the general compiler, but the system compiles to a '
        f'{type(problem).__name__} of the {problem.form} form.'
    )
    raise AnalysisError(error_msg)
