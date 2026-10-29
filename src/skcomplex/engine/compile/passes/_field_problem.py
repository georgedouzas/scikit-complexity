"""Pass that adds the stable step, the linear terms, the finite volumes, and the fixed nodes to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ...problems import ProblemSpec
from .._base import BasePass


class FieldProblemPass(BasePass):
    """Pass that adds the stable step, the linear terms, the finite volumes, and the fixed nodes to the problem."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'limits', 'volume_limits', 'linear', 'volumes', 'constrained')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the discretizations of the fields to the problem."""
        compilation = self.compilation_
        problem = cast('ProblemSpec', compilation.problem)
        indices, values = compilation.constrained
        initial_state = problem.initial_state.copy()
        initial_state[indices] = values
        compilation.problem = replace(
            problem,
            initial_state=initial_state,
            stable_step=min(compilation.limits, default=None),
            diffusive_limits=tuple(compilation.volume_limits),
            linear=tuple(compilation.linear),
            volumes=compilation.volumes,
            constrained=indices,
        )
