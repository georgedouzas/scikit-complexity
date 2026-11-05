"""Pass that expands the flows of components into laws of other kinds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from .._base import BasePass
from .._compilation import Compilation
from .._flows import TransferLowering


class ExpandFlowsPass(BasePass):
    """Pass that expands the flows of components into laws of other kinds."""

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
        collected = Compilation._required(compilation.collected, 'collected')
        expanded = flows.run([(spec, laws) for spec, laws, _ in collected]).lowered_
        compilation.collected = [
            (spec, laws, block) for (spec, laws), (_, _, block) in zip(expanded, collected, strict=True)
        ]
        compilation.transfer = flows._state()
