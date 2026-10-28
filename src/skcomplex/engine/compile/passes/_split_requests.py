"""Pass that sets the update laws that request topology changes apart from the laws lowered into kernels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import BaseLaw, UpdateLaw
from ....specs import KernelSpec
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._compilation import BlockLaws, Laws


class SplitRequestsPass(BasePass):
    """Pass that sets the update laws that request topology changes apart from the laws lowered into kernels."""

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
