"""Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._flows import TransferLowering


class LowerFlowsPass(BasePass):
    """Pass that checks the conserved quantities and lowers flows and delays into the laws of every spec, in place."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout', 'transfer')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered', 'transfer_network')

    def _run(self, inputs: object = None) -> None:
        """Lower the flows and their delays."""
        compilation = self.compilation_
        flows = TransferLowering().check().prepare(compilation)
        lowered = Compilation._required(compilation.lowered, 'lowered')
        compilation.transfer_network = flows._lower_flows([(spec, laws) for spec, laws, _ in lowered])
