"""Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._base import BasePass
from .._compilation import BlockLaws
from .._declarations import Declarations
from .._flows import TransferPass


class LowerFlowsPass(BasePass):
    """Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout', 'flows')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered', 'transfer_network')

    def _run(self, inputs: object = None) -> None:
        """Lower the flows and their delays."""
        compilation = self.compilation_
        flows = cast('TransferPass', compilation.flows)
        lowered = cast('BlockLaws', compilation.lowered)
        declarations = cast('Declarations', compilation.layout)
        compilation.transfer_network = flows.lower([(spec, laws) for spec, laws, _ in lowered], declarations)
