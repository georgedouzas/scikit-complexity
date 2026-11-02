"""Pass that turns the jumps that request topology changes into jumps of no state change, noting the requests."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.laws import BaseLaw, JumpLaw
from ....specs import TopologyRequestSpec
from ...problems import KernelSpec
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._compilation import Compilation, Laws


class RequestJumpsPass(BasePass):
    """Pass that turns the jumps that request topology changes into jumps of no state change, noting the requests."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('reduced',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('reduced', 'request_jumps')

    def _run(self, inputs: object = None) -> None:
        """Note the requests of the jumps."""
        compilation = self.compilation_
        compilation.reduced, compilation.request_jumps = self._set_request_jumps_apart(
            Compilation._required(compilation.reduced, 'reduced'),
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
