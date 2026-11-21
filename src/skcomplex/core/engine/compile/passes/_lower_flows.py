"""Check the conserved quantities and build the network of stocks and flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._flows import TransferLowering


class LowerFlowsPass(BasePass):
    """A pass that checks the conserved quantities and builds the network of flows.

    It checks that a conserved quantity, such as the water of a tub, changes only through flows, and records the stocks,
    the flows between them, and the delays of the flows.

    Read more in the user guide, under Extend the library.

    See Also:
        ExpandFlowsPass: Turns the flows into laws.
        FlowsProblemPass: Adds the network to the problem.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.engineering.processes import BathtubSystem
        >>> compilation = PassCompiler().prepare(BathtubSystem()).compilation_
        >>> [flow.path for flow in compilation.transfer_network.flows]
        ['tub.faucet', 'tub.drain']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout', 'transfer')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered', 'transfer_network')

    def _run(self, inputs: object = None) -> None:
        """Lower the flows and their delays."""
        compilation = self.compilation_
        flows = TransferLowering().prepare(compilation)
        lowered = Compilation._get_required(compilation.lowered, 'lowered')
        compilation.transfer_network = flows._lower_flows([(spec, laws) for spec, laws, _ in lowered])
