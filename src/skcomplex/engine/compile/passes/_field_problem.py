"""Pass that adds the stable step, the linear terms, the finite volumes, and the fixed nodes to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ...problems import Problem
from .._compilation import Compilation
from .._pipeline import BasePass


class FieldProblemPass(BasePass):
    """Pass that adds the stable step, the linear terms, the finite volumes, and the fixed nodes to the problem."""

    reads = ('problem', 'limits', 'volume_limits', 'linear', 'volumes', 'constrained')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the discretizations of the fields to the problem.

        Args:
            compilation:
                The compilation state.
        """
        problem = cast('Problem', compilation.problem)
        indices, values = compilation.facts['constrained']
        initial_state = problem.initial_state.copy()
        initial_state[indices] = values
        compilation.problem = replace(
            problem,
            initial_state=initial_state,
            stable_step=min(compilation.facts['limits'], default=None),
            diffusive_limits=tuple(compilation.facts['volume_limits']),
            linear=tuple(compilation.facts['linear']),
            volumes=compilation.facts['volumes'],
            constrained=indices,
        )
