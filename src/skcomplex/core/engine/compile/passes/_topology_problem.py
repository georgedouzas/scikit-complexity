"""Pass that adds the topology changes to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from .._base import BasePass
from .._compilation import Compilation


class TopologyProblemPass(BasePass):
    """A pass that adds the topology changes to the problem."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'topology')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the topology changes to the problem."""
        compilation = self.compilation_
        problem = Compilation._get_required(compilation.problem, 'problem')
        compilation.problem = replace(problem, topology=compilation.topology)
