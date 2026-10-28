"""Pass that compiles the topology changes, leaving out the compiled stepper loops when laws request them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from dataclasses import replace
from typing import Any, cast

from sage.all__sagemath_symbolics import SR

from ....core.errors import ValidationError
from ....core.laws import BaseLaw, JumpLaw, UpdateLaw
from ....core.spaces import NetworkSpace
from ....core.specs import TopologyRequestSpec
from ....core.types import Vector
from ...problems import (
    EdgeSearch,
    Entry,
    LawKernel,
    Layout,
    Lowered,
    Output,
    ReductionSchedule,
    VariableKind,
    declarations_of,
    declared_variables,
    default_of,
)
from ...problems._capacity_rows import CapacityRows
from ...problems._compiled_request import CompiledRequest
from ...problems._network_rows import NetworkRows
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._block import Block
from .._compilation import Laws
from .._declarations import _ACTIVE, ENDS, Declarations
from .._kernel_spec import KernelSpec
from ._specs import SpecsPass
from ._weight import _Weight


class TopologyFunctionsPass(BasePass):
    """Pass that compiles the topology changes, leaving out the compiled stepper loops when laws request them."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('functions', 'kernels', 'reduced_apart', 'layout', 'reductions', 'request_jumps')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('functions', 'topology')

    def _run(self, inputs: object = None) -> None:
        """Compile the topology changes of the networks."""
        compilation = self.compilation_
        topology = self._topology(
            cast('Lowered', compilation.kernels),
            cast('Laws', compilation.reduced_apart),
            cast('Declarations', compilation.layout),
            cast('ReductionSchedule', compilation.reductions),
            compilation.request_jumps,
        )
        if topology is not None and (topology.requests or topology.channels):
            cast('dict[str, Any]', compilation.functions)['accelerated'] = None
        compilation.topology = topology

    @staticmethod
    def _request_kernel(spec: KernelSpec, value: object, declarations: Declarations) -> LawKernel:
        """Return the kernel of the probability or the weight of a request for every row of its spec."""
        states, parameters = Layout(declarations.states), Layout(declarations.parameters)
        expression = SR(value)
        return LawKernel(
            spec.owner,
            declarations.time,
            spec.references,
            spec.rows,
            states,
            parameters,
            spec.pairs,
            [Output(None, expression)],
        ).check()

    @staticmethod
    def _compiled_request(
        spec: KernelSpec,
        request: TopologyRequestSpec,
        networks: dict[int, NetworkRows],
        weights: dict[int, LawKernel],
        declarations: Declarations,
    ) -> CompiledRequest:
        """Return a request of a spec with its network, its rows, and the kernel of its weights."""
        space, rows = TopologyFunctionsPass._request_network(spec, declarations)
        network = networks.get(id(space))
        if network is None:
            error_msg = f'{spec.owner} requests {request!r} on a network without variable groups on its nodes.'
            raise ValidationError(error_msg)
        if TopologyKernel._changes_edges(request) and request.layer not in network.edges:
            error_msg = (
                f'{spec.owner} requests {request!r}, which changes the edges of the layer {request.layer!r}, but the '
                'layer has no variable edge group. Add `GroupElement(..., on="edges", variable=True)` on the layer.'
            )
            raise ValidationError(error_msg)
        needs = {'remove-edge': 'edge', 'rewire': 'edge', 'add-edge': 'node', 'remove-node': 'node'}.get(request.kind)
        if needs is not None and needs != rows:
            error_msg = f'{spec.owner} requests {request!r}, which runs for every {needs}, from laws of every {rows}.'
            raise ValidationError(error_msg)
        return CompiledRequest(request, spec.owner, network, rows, toward=weights.get(id(request)))

    @staticmethod
    def _is_weight(law: BaseLaw) -> bool:
        """Check whether a law sets the weight of the candidate nodes of a request."""
        return isinstance(next(iter(law._get_targets()), None), _Weight)

    @staticmethod
    def _topology(
        lowered: Lowered,
        apart: Laws,
        declarations: Declarations,
        reductions: ReductionSchedule,
        jumps: dict[int, tuple[KernelSpec, TopologyRequestSpec]],
    ) -> TopologyKernel | None:
        """Return the topology pass of the variable groups of networks and the requests of laws, or `None` if none."""
        networks = TopologyFunctionsPass._networks(declarations)
        if not networks:
            return None
        towards = [(spec, laws) for spec, laws in apart if all(TopologyFunctionsPass._is_weight(law) for law in laws)]
        updates = [
            (spec, laws) for spec, laws in apart if not all(TopologyFunctionsPass._is_weight(law) for law in laws)
        ]
        weights = {
            id(law._get_targets()[0].request): TopologyFunctionsPass._request_kernel(spec, law.value, declarations)
            for spec, laws in towards
            for law in laws
            if isinstance(law, UpdateLaw)
        }
        requests = []
        for spec, laws in updates:
            for law in laws:
                request = cast('TopologyRequestSpec', TopologyKernel._request_of(law))
                item = TopologyFunctionsPass._compiled_request(spec, request, networks, weights, declarations)
                requests.append(
                    replace(
                        item,
                        condition=TopologyFunctionsPass._request_kernel(
                            spec,
                            cast('UpdateLaw', law).value,
                            declarations,
                        ),
                    ),
                )
        channels = {}
        index = 0
        for kernel in lowered.kernels[JumpLaw]:
            for output in kernel.outputs:
                _, law = lowered.laws[id(output)]
                found = jumps.get(id(law))
                if found is not None:
                    channels[index] = TopologyFunctionsPass._compiled_request(*found, networks, weights, declarations)
                index += 1
        return TopologyKernel(tuple(networks.values()), tuple(requests), channels, reductions).check()

    @staticmethod
    def _capacity_rows(block: Block, declarations: Declarations) -> CapacityRows:
        """Return the rows of a variable group with the values of one new member.

        Args:
            block:
                The block of the variable group.

            declarations:
                The declarations of the compilation.

        Returns:
            rows:
                The capacity rows of the group.
        """
        node = block.node
        element = node.component_type
        states = declarations_of(element, VariableKind.STATE)
        variables: dict[str, tuple[Entry, bool]] = {}
        defaults: dict[str, Vector] = {}
        for name, entry in block.entries.items():
            if name in {_ACTIVE, *ENDS}:
                continue
            variables[name] = (entry, name in states)
            default = default_of(element, name) if name in declared_variables(element) else 0.0
            default = 0.0 if default is dataclasses.MISSING else default
            defaults[name] = Declarations._coerce_value(entry.name, default, entry.member_shape, entry.domain, None)
        source, target = block.entries.get(ENDS[0]), block.entries.get(ENDS[1])
        return CapacityRows(node.path, variables, defaults, block.entries[_ACTIVE], source, target)

    @staticmethod
    def _networks(declarations: Declarations) -> dict[int, NetworkRows]:
        """Return the variable groups of every network whose groups are variable, by the identity of the space."""
        spaces = {id(block.node.space): block.node.space for block in declarations.capacity.values()}
        networks = {}
        for key, space in spaces.items():
            nodes = tuple(
                TopologyFunctionsPass._capacity_rows(block, declarations)
                for block in declarations.capacity.values()
                if block.node.space is space and getattr(block.node.component, 'on', None) == 'sites'
            )
            if not nodes:
                continue
            sizes = {group.active.rows for group in nodes}
            if len(sizes) > 1:
                error_msg = (
                    f'The variable groups on the nodes of one network, {", ".join(group.path for group in nodes)}, '
                    'have different capacities. Give them one capacity.'
                )
                raise ValidationError(error_msg)
            edges = {
                layer: TopologyFunctionsPass._capacity_rows(block, declarations)
                for layer, block in declarations.changing_edges(space).items()
            }
            directed = {layer: cast('NetworkSpace', space).graphs_[layer].is_directed() for layer in edges}
            networks[key] = NetworkRows(nodes, edges, directed)
        return networks

    @staticmethod
    def _request_network(spec: KernelSpec, declarations: Declarations) -> tuple[object, str]:
        """Return the network a request of a spec changes and the rows the spec runs on."""
        if isinstance(spec.pairs, EdgeSearch):
            nodes = declarations.blocks[cast('str', spec.roles['source'])]
            return SpecsPass._group_space(nodes), 'edge'
        if spec.group in declarations.capacity:
            return SpecsPass._group_space(declarations.capacity[spec.group]), 'node'
        if spec.rows == 1:
            spaces = {id(block.node.space): block.node.space for block in declarations.capacity.values()}
            if len(spaces) == 1:
                return next(iter(spaces.values())), 'single'
        error_msg = (
            f'{spec.owner} requests topology changes, but it runs on neither the members of a variable group on the '
            'nodes or edges of a network nor once for the one network whose groups are variable. Give the groups of '
            'the network `variable=True`.'
        )
        raise ValidationError(error_msg)
