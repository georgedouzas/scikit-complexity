"""Pass that sets apart the weight of the candidate nodes of every request, read through reductions like laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, cast

from sage.all__sagemath_symbolics import SR

from ....core.laws import BaseLaw, UpdateLaw
from ....errors import ValidationError
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._block import Block
from .._compilation import Laws
from .._declarations import Declarations
from .._kernel_spec import KernelSpec
from ._specs import SpecsPass
from ._topology_functions import TopologyFunctionsPass
from ._weight import _Weight


class TowardsPass(BasePass):
    """Pass that sets apart the weight of the candidate nodes of every request, read through reductions like laws."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn_apart', 'drawn', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn_apart',)

    def _run(self, inputs: object = None) -> None:
        """Set the weights of the requests apart."""
        compilation = self.compilation_
        apart = cast('Laws', compilation.drawn_apart)
        towards = self._toward_specs(
            [*cast('Laws', compilation.drawn), *apart],
            cast('Declarations', compilation.layout),
        )
        compilation.drawn_apart = [*apart, *towards]

    @staticmethod
    def _node_block(space: object, declarations: Declarations) -> Block:
        """Return the first variable group on the nodes of a network."""
        for block in declarations.capacity.values():
            if getattr(block.node.component, 'on', None) == 'sites' and block.node.space is space:
                return block
        error_msg = 'A topology request changes a network without a variable group on its nodes. Add one.'
        raise ValidationError(error_msg)

    @staticmethod
    def _toward_specs(specs: list[tuple[KernelSpec, list[BaseLaw]]], declarations: Declarations) -> list[Any]:
        """Return the specs of the nodes of every request with a weight, each with the law of the weight to lower."""
        towards: list[tuple[KernelSpec, list[BaseLaw]]] = []
        for spec, laws in specs:
            for law in laws:
                request = TopologyKernel._request_of(law)
                if request is None or request.toward is None:
                    continue
                space, _ = TopologyFunctionsPass._request_network(spec, declarations)
                nodes = TowardsPass._node_block(space, declarations)
                node_spec = SpecsPass._member_spec(nodes, declarations.time)
                towards.append((node_spec, [UpdateLaw(_Weight(request), SR(request.toward(node_spec.namespace)))]))
        return towards
