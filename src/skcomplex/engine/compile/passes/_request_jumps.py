"""Pass that turns the jumps that request topology changes into jumps of no state change, noting the requests."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.laws import BaseLaw, JumpLaw
from ....core.specs import TopologyRequestSpec
from ...problems._topology_kernel import TopologyKernel
from .._compilation import Compilation, Laws
from .._kernel_spec import KernelSpec
from .._pipeline import BasePass


class RequestJumpsPass(BasePass):
    """Pass that turns the jumps that request topology changes into jumps of no state change, noting the requests."""

    reads = ('reduced',)
    writes = ('reduced', 'request_jumps')

    def _run(self, compilation: Compilation) -> None:
        """Note the requests of the jumps.

        Args:
            compilation:
                The compilation state.
        """
        compilation.reduced, compilation.facts['request_jumps'] = self._set_request_jumps_apart(
            cast('Laws', compilation.reduced),
        )

    @staticmethod
    def _set_request_jumps_apart(
        specs: Laws,
    ) -> tuple[Laws, dict[int, tuple[KernelSpec, TopologyRequestSpec]]]:
        """Return laws whose jumps that request topology changes change no state, with the request of each such jump."""
        requests: dict[int, tuple[KernelSpec, TopologyRequestSpec]] = {}
        result: Laws = []
        for spec, laws in specs:
            kept: list[BaseLaw] = []
            for law in laws:
                request = TopologyKernel._request_of(law) if isinstance(law, JumpLaw) else None
                if not isinstance(law, JumpLaw) or request is None:
                    kept.append(law)
                    continue
                jump = JumpLaw(law.rate, {})
                requests[id(jump)] = (spec, request)
                kept.append(jump)
            result.append((spec, kept))
        return result, requests
