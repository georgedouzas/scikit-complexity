"""Pass that sets apart the weights and scores of the candidate nodes of requests.

They are read like laws, for every request.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from sage.all__sagemath_symbolics import SR

from ....components.laws import BaseLaw, UpdateLaw
from ....errors import ValidationError
from ....specs._weight_spec import _WeightSpec
from ...problems._block_spec import BlockSpec
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._kernel_spec import KernelSpec
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._compilation import Compilation
from ._specs import SpecsPass
from ._topology_functions import TopologyFunctionsPass


class TowardsPass(BasePass):
    """Pass that sets apart the weights and scores of the candidate nodes of requests.

    They are read like laws, for every request.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn_apart', 'drawn', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn_apart',)

    def _run(self, inputs: object = None) -> None:
        """Set the weights of the requests apart."""
        compilation = self.compilation_
        apart = Compilation._get_required(compilation.drawn_apart, 'drawn_apart')
        towards = self._get_toward_specs(
            [*Compilation._get_required(compilation.drawn, 'drawn'), *apart],
            Compilation._get_required(compilation.layout, 'layout'),
        )
        compilation.drawn_apart = [*apart, *towards]

    @staticmethod
    def _get_node_block(space: object, declarations: DeclarationsSpec) -> BlockSpec:
        """Return the first variable group on the nodes of a network."""
        for block in declarations.capacity.values():
            if getattr(block.node.component, 'on', None) == 'sites' and block.node.space is space:
                return block
        error_msg = 'A topology request changes a network without a variable group on its nodes. Add one.'
        raise ValidationError(error_msg)

    @staticmethod
    def _get_toward_specs(
        specs: list[tuple[KernelSpec, list[BaseLaw]]],
        declarations: DeclarationsSpec,
    ) -> list[tuple[KernelSpec, list[BaseLaw]]]:
        """Return the specs of the nodes of every request with a weight or a score.

        Each comes with the law to lower.
        """
        towards: list[tuple[KernelSpec, list[BaseLaw]]] = []
        for spec, laws in specs:
            for law in laws:
                request = TopologyKernel._request_of(law)
                if request is None or (request.toward is None and request.score is None):
                    continue
                space, _ = TopologyFunctionsPass._request_network(spec, declarations)
                nodes = TowardsPass._get_node_block(space, declarations)
                node_spec = SpecsPass._get_member_spec(nodes, declarations.time)
                references = dict(node_spec.references)
                namespace = SpecsPass._copy_with_source(nodes, node_spec.namespace, references)
                weights = [
                    UpdateLaw(_WeightSpec(request, score=part == 'score'), SR(function(namespace)))
                    for part, function in (('toward', request.toward), ('score', request.score))
                    if function is not None
                ]
                towards.append((replace(node_spec, references=references), weights))
        return towards
