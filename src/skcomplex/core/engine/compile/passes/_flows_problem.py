"""Add the network of stocks and flows to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from .._base import BasePass
from .._compilation import Compilation


class FlowsProblemPass(BasePass):
    """A pass that adds the network of stocks and flows to the problem.

    A run then records the amount every flow moves, such as the water that leaves a tub through its drain.

    Read more in the user guide, under Extend the library.

    See Also:
        LowerFlowsPass: Builds the network.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.engineering.processes import BathtubSystem
        >>> compilation = PassCompiler().prepare(BathtubSystem()).run().compilation_
        >>> compilation.problem.transfer_network.quantities
        {'water': ('tub.water',)}
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
            Compilation._get_required(compilation.problem, 'problem'),
            transfer_network=Compilation._get_required(compilation.transfer_network, 'transfer_network'),
        )
