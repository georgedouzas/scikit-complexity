"""Pass that adds the transfer network to the problem.

It adds the quantities, flows, and records of the network.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from .._base import BasePass
from .._compilation import Compilation


class FlowsProblemPass(BasePass):
    """Pass that adds the transfer network to the problem.

    It adds the quantities, flows, and records of the network.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'transfer_network')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the transfer network to the problem."""
        compilation = self.compilation_
        compilation.problem = replace(
            Compilation._required(compilation.problem, 'problem'),
            transfer_network=Compilation._required(compilation.transfer_network, 'transfer_network'),
        )
