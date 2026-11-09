"""Share the small helpers of the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import cache
from importlib import import_module
from types import ModuleType

from ..components.systems import CompositeSystem
from ..engine.compile._pass_compiler import PassCompiler
from ..engine.problems import BaseProblemSpec, ProblemSpec
from ..errors import AnalysisError


@cache
def load_module(name: str) -> ModuleType:
    """Return a module, importing it on the first call.

    Args:
        name:
            The dotted name of the module.

    Returns:
        module:
            The imported module.

    Examples:
        >>> from skcomplex.core.analysis._utils import load_module
        >>> load_module('math').sqrt(4.0)
        2.0
    """
    return import_module(name)


def compile_problem(system: CompositeSystem | BaseProblemSpec, purpose: str) -> ProblemSpec:
    """Return the general compiler's problem of a system, or the problem itself.

    It serves work that reads its fields.

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
    """
    problem = system if isinstance(system, BaseProblemSpec) else PassCompiler().prepare(system).run().problem_
    if isinstance(problem, ProblemSpec):
        return problem
    error_msg = (
        f'{purpose} reads the problem of the general compiler, but the system compiles to a '
        f'{type(problem).__name__} of the {problem.form} form.'
    )
    raise AnalysisError(error_msg)
