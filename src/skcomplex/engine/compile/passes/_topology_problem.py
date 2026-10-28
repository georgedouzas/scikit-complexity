"""Pass that adds the topology changes to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ...problems import Problem
from .._base import BasePass


class TopologyProblemPass(BasePass):
    """Pass that adds the topology changes to the problem."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'topology')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the topology changes to the problem."""
        compilation = self.compilation_
        compilation.problem = replace(cast('Problem', compilation.problem), topology=compilation.topology)
