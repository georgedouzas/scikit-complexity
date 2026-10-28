"""Pass that adds the topology changes to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ...problems import Problem
from .._compilation import Compilation
from .._pipeline import BasePass


class TopologyProblemPass(BasePass):
    """Pass that adds the topology changes to the problem."""

    reads = ('problem', 'topology')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the topology changes to the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.problem = replace(cast('Problem', compilation.problem), topology=compilation.facts['topology'])
