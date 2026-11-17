"""Define the base of the solvers that read the objective of a static system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np
from sage.all__sagemath_symbolics import Expression, diff

from ...errors import SolverError
from ...types import Array
from .._utils import symbols_of
from ..compile._observables import ObservableEvaluator
from ..problems import ProblemSpec
from ..problems._expression_printer import ExpressionPrinter
from ._solver import Solver


class BaseObjectiveSolver(Solver):
    """An abstract solver of the objective of a static system.

    It turns the sum of the objective laws into a function that gives the objective and its gradient over the target
    states. A subclass decides what to do with it: minimise it once, follow its steepest descent, or sample states with
    its Boltzmann weight.
    """

    def _get_objective_function(self, problem: ProblemSpec) -> Callable[[Array, Array], tuple[float, Array]]:
        """Return the function that gives the objective and its gradient."""
        objective = problem.objective
        if not isinstance(objective, Expression):
            error_msg = f'{type(self).__name__} minimises an objective, but the problem has no objective law.'
            raise SolverError(error_msg)
        sources = ObservableEvaluator(problem)._get_scalar_sources()
        positions = {index: item for item, (vector, index) in sources.items() if vector == 'x'}
        missing = [int(index) for index in problem.objective_targets if int(index) not in positions]
        if missing:
            error_msg = (
                f'The objective is minimised over the state positions {missing}, which belong to group members. '
                'Minimise over the states of single elements.'
            )
            raise SolverError(error_msg)
        targets = [positions[int(index)] for index in problem.objective_targets]
        expressions = [objective, *(diff(objective, item) for item in targets)]
        arguments = sorted(symbols_of(expressions) - {problem.time}, key=str)
        unknown = [item for item in arguments if item not in sources]
        if unknown:
            error_msg = f'The objective reads {unknown}, which are no states or parameters of single elements.'
            raise SolverError(error_msg)
        printed = ExpressionPrinter(time=problem.time, compiled=True).prepare(arguments).run(expressions).code_
        if not callable(printed):
            error_msg = f'A compiled printer returns a function, not a {type(printed).__name__}.'
            raise SolverError(error_msg)
        reads = [sources[item] for item in arguments]

        def evaluate(x: Array, p: Array) -> tuple[float, Array]:
            values = printed(0.0, *(x[index] if vector == 'x' else p[index] for vector, index in reads))
            return float(np.real(values[0])), np.real(np.asarray(values[1:], dtype=complex)).astype(float)

        return evaluate
