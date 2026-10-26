"""Report the systems whose run an optimal control solver cannot optimise, before compilation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...core.base import SolverError, TimeKind, ValidationError
from ...engine.compile import Compilation, Issue, declared_times
from ._optimal import OptimalControlSolver

OBJECTIVE_TIMES = frozenset({TimeKind.STATIC, TimeKind.CONTINUOUS})


def objective_issues(compilation: Compilation) -> list[Issue]:
    """Return the invalid objective of an optimal control solver, and the subsystems of a time it cannot optimise.

    Args:
        compilation:
            The compilation state after planning.

    Returns:
        issues:
            The issues.
    """
    solver = compilation.solver
    if not isinstance(solver, OptimalControlSolver):
        return []
    system = compilation.system
    name = type(solver).__name__
    issues = []
    try:
        solver.check()
    except (SolverError, ValidationError) as error:
        fix = "Pass `objective`, and `sense` as 'min' or 'max'."
        issues.append(Issue('objective-argument', system.label, str(error), fix, error=type(error)))
    for path, time in declared_times(system).items():
        if time is None or time in OBJECTIVE_TIMES:
            continue
        message = (
            f'The objective of {name} needs static or continuous time, but {path or system.label} declares {time} '
            'time.'
        )
        fix = f'Declare static or continuous time, or run the system with a solver of {time} time.'
        issues.append(Issue('objective-time', path or system.label, message, fix, error=SolverError))
    return issues
