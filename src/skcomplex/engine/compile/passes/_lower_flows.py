"""Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._compilation import BlockLaws, Compilation
from .._declarations import Declarations
from .._flows import TransferPass
from .._pipeline import BasePass


class LowerFlowsPass(BasePass):
    """Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

    reads = ('lowered', 'layout', 'flows')
    writes = ('lowered', 'transfer_network')

    def _run(self, compilation: Compilation) -> None:
        """Lower the flows and their delays.

        Args:
            compilation:
                The compilation state.
        """
        flows = cast('TransferPass', compilation.facts['flows'])
        lowered = cast('BlockLaws', compilation.lowered)
        declarations = cast('Declarations', compilation.layout)
        compilation.facts['transfer_network'] = flows.lower([(spec, laws) for spec, laws, _ in lowered], declarations)
