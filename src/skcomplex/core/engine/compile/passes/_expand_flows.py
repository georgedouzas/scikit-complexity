"""Turn the flows of components into laws of other kinds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._flows import TransferLowering


class ExpandFlowsPass(BasePass):
    """A pass that turns the flows of components into laws of other kinds.

    A flow moves an amount from a source to a sink, such as water from a faucet into a tub, and the pass writes the laws
    that take it from the one and add it to the other.

    Read more in the user guide, under Extend the library.

    See Also:
        TransferLowering: Turns the flows into laws.
        LowerFlowsPass: Finishes the work.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.engineering.processes import BathtubSystem
        >>> compilation = PassCompiler().prepare(BathtubSystem()).compilation_
        >>> len(compilation.transfer.flows)
        2
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected', 'transfer')

    def _run(self, inputs: object = None) -> None:
        """Expand the flows of the collected laws."""
        compilation = self.compilation_
        flows = TransferLowering().prepare(compilation)
        collected = Compilation._get_required(compilation.collected, 'collected')
        expanded = flows.run([(spec, laws) for spec, laws, _ in collected]).lowered_
        compilation.collected = [
            (spec, laws, block) for (spec, laws), (_, _, block) in zip(expanded, collected, strict=True)
        ]
        compilation.transfer = flows._get_state()
