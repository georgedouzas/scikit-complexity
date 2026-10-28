"""Pass that adds the quantities, flows, and records of the transfer network to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ...problems import Problem, TransferNetwork
from .._compilation import Compilation
from .._pipeline import BasePass


class FlowsProblemPass(BasePass):
    """Pass that adds the quantities, flows, and records of the transfer network to the problem."""

    reads = ('problem', 'transfer_network')
    writes = ('problem',)

    def _run(self, compilation: Compilation) -> None:
        """Add the transfer network to the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.problem = replace(
            cast('Problem', compilation.problem),
            transfer_network=cast('TransferNetwork', compilation.facts['transfer_network']),
        )
