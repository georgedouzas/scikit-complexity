"""Set apart the update laws that change the shape of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws import BaseLaw, UpdateLaw
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._compilation import BlockLaws, Compilation


class SplitRequestsPass(BasePass):
    """A pass that sets apart the update laws that change the shape of a network.

    An update law may ask for a new node or edge, such as a newcomer to a network who links to popular nodes. The pass
    sets such laws apart, since they change the network and not a value.

    Read more in the user guide, under Extend the library.

    See Also:
        TopologyFunctionsPass: Compiles the changes of the network.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.mathematics.networks import BarabasiAlbertSystem
        >>> compilation = PassCompiler().prepare(BarabasiAlbertSystem()).compilation_
        >>> [spec.owner for spec, _ in compilation.apart]
        ['growth']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected', 'apart')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected', 'apart')

    def _run(self, inputs: object = None) -> None:
        """Set the update laws that request topology changes apart."""
        compilation = self.compilation_
        collected: BlockLaws = []
        apart = Compilation._get_required(compilation.apart, 'apart')
        for spec, laws, block in Compilation._get_required(compilation.collected, 'collected'):
            kept, updates = self._split_requests(laws)
            collected.append((spec, kept, block))
            if updates:
                apart.append((spec, updates))
        compilation.collected = collected

    @staticmethod
    def _split_requests(laws: list[BaseLaw]) -> tuple[list[BaseLaw], list[BaseLaw]]:
        """Return the laws of a spec without its topology request updates, and those.

        They are the update laws that request topology changes.
        """
        updates: list[BaseLaw] = [
            law for law in laws if isinstance(law, UpdateLaw) and TopologyKernel._request_of(law) is not None
        ]
        kept = [law for law in laws if not any(law is item for item in updates)]
        return kept, updates
