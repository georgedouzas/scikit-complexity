"""Pass that adds the quantities, flows, and records of the transfer network to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from ....specs import ProblemSpec, TransferNetworkSpec
from .._base import BasePass


class FlowsProblemPass(BasePass):
    """Pass that adds the quantities, flows, and records of the transfer network to the problem."""

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
            cast('ProblemSpec', compilation.problem),
            transfer_network=cast('TransferNetworkSpec', compilation.transfer_network),
        )
