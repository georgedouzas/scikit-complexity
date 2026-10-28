"""Pass that sets the update laws that request topology changes apart from the laws lowered into kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import BaseLaw, UpdateLaw
from ...problems._topology_kernel import TopologyKernel
from .._compilation import BlockLaws, Compilation, Laws
from .._kernel_spec import KernelSpec
from .._pipeline import BasePass


class SplitRequestsPass(BasePass):
    """Pass that sets the update laws that request topology changes apart from the laws lowered into kernels."""

    reads = ('collected', 'apart')
    writes = ('collected', 'apart')

    def _run(self, compilation: Compilation) -> None:
        """Set the update laws that request topology changes apart.

        Args:
            compilation:
                The compilation state.
        """
        collected: BlockLaws = []
        apart = cast('Laws', compilation.apart)
        for spec, laws, block in cast('BlockLaws', compilation.collected):
            kept, updates = self._split_requests(spec, laws)
            collected.append((spec, kept, block))
            if updates:
                apart.append((spec, updates))
        compilation.collected = collected

    @staticmethod
    def _split_requests(spec: KernelSpec, laws: list[BaseLaw]) -> tuple[list[BaseLaw], list[BaseLaw]]:
        """Return the laws of a spec without its update laws that request topology changes, and those update laws."""
        updates: list[BaseLaw] = [
            law for law in laws if isinstance(law, UpdateLaw) and TopologyKernel._request_of(law) is not None
        ]
        kept = [law for law in laws if not any(law is item for item in updates)]
        return kept, updates
