"""Pass that sets the topology request update laws apart from the kernel laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....components.laws import BaseLaw, UpdateLaw
from ...problems._kernel_spec import KernelSpec
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._compilation import BlockLaws, Compilation


class SplitRequestsPass(BasePass):
    """A pass that sets the topology request update laws apart from the kernel laws.

    They are the update laws that request topology changes, and the kernel laws are those lowered into kernels.
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
            kept, updates = self._split_requests(spec, laws)
            collected.append((spec, kept, block))
            if updates:
                apart.append((spec, updates))
        compilation.collected = collected

    @staticmethod
    def _split_requests(spec: KernelSpec, laws: list[BaseLaw]) -> tuple[list[BaseLaw], list[BaseLaw]]:
        """Return the laws of a spec without its topology request updates, and those.

        They are the update laws that request topology changes.
        """
        updates: list[BaseLaw] = [
            law for law in laws if isinstance(law, UpdateLaw) and TopologyKernel._request_of(law) is not None
        ]
        kept = [law for law in laws if not any(law is item for item in updates)]
        return kept, updates
