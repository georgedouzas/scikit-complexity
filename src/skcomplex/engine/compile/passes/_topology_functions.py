"""Pass that compiles the topology changes, leaving out the compiled stepper loops when laws request them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from dataclasses import replace

from sage.all__sagemath_symbolics import SR

from ....core.laws import BaseLaw, JumpLaw, UpdateLaw
from ....core.spaces import NetworkSpace
from ....errors import PipelineError, ValidationError
from ....specs import CapacityRowsSpec, EntrySpec, LayoutSpec, NetworkRowsSpec, TopologyRequestSpec
from ....specs._weight_spec import _WeightSpec
from ....types import Array, VariableKind
from ..._utils import declarations_of, declared_variables, default_of
from ...problems import (
    BlockSpec,
    CompiledRequestSpec,
    DeclarationsSpec,
    KernelSpec,
    LoweredKernelsSpec,
    OutputSpec,
    ReductionScheduleSpec,
)
from ...problems._declarations_spec import _ACTIVE, ENDS, MEMBERS
from ...problems._edge_search import EdgeSearch
from ...problems._law_kernel import LawKernel
from ...problems._topology_kernel import TopologyKernel
from .._base import BasePass
from .._compilation import Compilation, Laws
from ._layout import LayoutPass
from ._specs import SpecsPass


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
        functions, jumps = compilation.functions, compilation.request_jumps
        if functions is None or jumps is None:
            error_msg = 'TopologyFunctionsPass reads the numeric functions and the request jumps, which no pass wrote.'
            raise PipelineError(error_msg)
        topology = self._topology(
            Compilation._required(compilation.kernels, 'kernels'),
            Compilation._required(compilation.reduced_apart, 'reduced_apart'),
            Compilation._required(compilation.layout, 'layout'),
            Compilation._required(compilation.reductions, 'reductions'),
            jumps,
        )
        if topology is not None and (topology.requests or topology.channels):
            compilation.functions = replace(functions, accelerated=None)
        compilation.topology = topology

    @staticmethod
    def _request_kernel(spec: KernelSpec, value: object, declarations: DeclarationsSpec) -> LawKernel:
        """Return the kernel of the probability or the weight of a request for every row of its spec."""
        states, parameters = LayoutSpec(tuple(declarations.states)), LayoutSpec(tuple(declarations.parameters))
        expression = SR(value)
        return (
            LawKernel(
                spec.owner,
                declarations.time,
                spec.references,
                spec.rows,
                states,
                layout_parameters=parameters,
                pairs=spec.pairs,
                outputs=[OutputSpec(None, expression)],
            )
            .check()
            .prepare()
        )

    @staticmethod
    def _compiled_request(
        spec: KernelSpec,
        request: TopologyRequestSpec,
        networks: dict[int, NetworkRowsSpec],
        weights: dict[int, LawKernel],
        declarations: DeclarationsSpec,
    ) -> CompiledRequestSpec:
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
        if request.hyperedge is not None and network.hyperedges is None:
            error_msg = (
                f'{spec.owner} requests {request!r}, which forms a hyperedge, but the network has no variable group '
                'on its hyperedges. Add `GroupElement(..., on="hyperedges", variable=True)`.'
            )
            raise ValidationError(error_msg)
        if request.hyperedge is not None and network.hyperedges is not None:
            size = network.hyperedges.members.columns if network.hyperedges.members is not None else 0
            if size != request.hyperedge:
                error_msg = (
                    f'{spec.owner} requests {request!r}, which forms hyperedges of size {request.hyperedge}, but the '
                    f'hyperedges of the network have size {size}. Give the request the size of the hyperedges.'
                )
                raise ValidationError(error_msg)
        needs = {'remove-edge': 'edge', 'rewire': 'edge', 'add-edge': 'node', 'remove-node': 'node'}.get(request.kind)
        if needs is not None and needs != rows:
            error_msg = f'{spec.owner} requests {request!r}, which runs for every {needs}, from laws of every {rows}.'
            raise ValidationError(error_msg)
        return CompiledRequestSpec(request, spec.owner, network, rows, toward=weights.get(id(request)))

    @staticmethod
    def _is_weight(law: BaseLaw) -> bool:
        """Check whether a law sets the weight of the candidate nodes of a request."""
        return isinstance(next(iter(law._get_targets()), None), _WeightSpec)

    @staticmethod
    def _topology(
        lowered: LoweredKernelsSpec,
        apart: Laws,
        declarations: DeclarationsSpec,
        reductions: ReductionScheduleSpec,
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
            id(target.request): TopologyFunctionsPass._request_kernel(spec, law.value, declarations)
            for spec, laws in towards
            for law in laws
            if isinstance(law, UpdateLaw) and isinstance(target := law._get_targets()[0], _WeightSpec)
        }
        requests = []
        for spec, laws in updates:
            for law in laws:
                request = TopologyKernel._request_of(law)
                if request is None or not isinstance(law, UpdateLaw):
                    error_msg = f'A law of {spec.owner} set apart for the networks requests no topology change.'
                    raise PipelineError(error_msg)
                item = TopologyFunctionsPass._compiled_request(spec, request, networks, weights, declarations)
                requests.append(
                    replace(item, condition=TopologyFunctionsPass._request_kernel(spec, law.value, declarations)),
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
        return TopologyKernel(tuple(networks.values()), tuple(requests), channels, reductions).check().prepare()

    @staticmethod
    def _capacity_rows(block: BlockSpec, declarations: DeclarationsSpec) -> CapacityRowsSpec:
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
        variables: dict[str, tuple[EntrySpec, bool]] = {}
        defaults: dict[str, Array] = {}
        for name, entry in block.entries.items():
            if name in {_ACTIVE, *ENDS, MEMBERS}:
                continue
            variables[name] = (entry, name in states)
            default = default_of(element, name) if name in declared_variables(element) else 0.0
            default = 0.0 if default is dataclasses.MISSING else default
            defaults[name] = LayoutPass._coerce_value(entry.name, default, entry.member_shape, entry.domain, None)
        source, target, members = (block.entries.get(name) for name in (*ENDS, MEMBERS))
        return CapacityRowsSpec(node.path, variables, defaults, block.entries[_ACTIVE], source, target, members)

    @staticmethod
    def _networks(declarations: DeclarationsSpec) -> dict[int, NetworkRowsSpec]:
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
            if not isinstance(space, NetworkSpace):
                error_msg = f'The variable groups {", ".join(group.path for group in nodes)} are not on a network.'
                raise ValidationError(error_msg)
            directed = {layer: space.graphs_[layer].is_directed() for layer in edges}
            hyperedges = next(
                (
                    TopologyFunctionsPass._capacity_rows(block, declarations)
                    for block in declarations.capacity.values()
                    if block.node.space is space and getattr(block.node.component, 'on', None) == 'hyperedges'
                ),
                None,
            )
            networks[key] = NetworkRowsSpec(nodes, edges, directed, hyperedges)
        return networks

    @staticmethod
    def _request_network(spec: KernelSpec, declarations: DeclarationsSpec) -> tuple[object, str]:
        """Return the network a request of a spec changes and the rows the spec runs on."""
        if isinstance(spec.pairs, EdgeSearch):
            source = spec.roles[declarations.blocks[spec.owner].node.component._get_role('source')]
            if not isinstance(source, str):
                error_msg = f'The source role of {spec.owner} binds several groups, not the nodes of one network.'
                raise ValidationError(error_msg)
            nodes = declarations.blocks[source]
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
