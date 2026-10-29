"""Pass that adds the stable step, the linear terms, the finite volumes, and the fixed nodes to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from ....errors import PipelineError
from .._base import BasePass
from .._compilation import Compilation


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
        problem = Compilation._required(compilation.problem, 'problem')
        constrained, limits, volume_limits = compilation.constrained, compilation.limits, compilation.volume_limits
        linear, volumes = compilation.linear, compilation.volumes
        if constrained is None or limits is None or volume_limits is None or linear is None or volumes is None:
            error_msg = 'FieldProblemPass reads the fields of the field functions, which no pass wrote.'
            raise PipelineError(error_msg)
        indices, values = constrained
        initial_state = problem.initial_state.copy()
        initial_state[indices] = values
        compilation.problem = replace(
            problem,
            initial_state=initial_state,
            stable_step=min(limits, default=None),
            diffusive_limits=tuple(volume_limits),
            linear=tuple(linear),
            volumes=volumes,
            constrained=indices,
        )
