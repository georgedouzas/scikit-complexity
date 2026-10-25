"""Compile the topology requests of laws, which add and remove the nodes and edges of variable networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, replace
from typing import Any, cast

from sage.all__sagemath_symbolics import SR

from ...core.base import BaseLaw, Layout, ValidationError
from ...core.laws import JumpLaw, UpdateLaw
from ...core.space import NetworkSpace
from ...engine.compile import (
    BasePass,
    Block,
    BlockLaws,
    Compilation,
    Declarations,
    EdgeSearch,
    Kernel,
    KernelSpec,
    Laws,
    Lowered,
    Output,
    Problem,
    ReductionPass,
    capacity_rows,
    group_space,
    mask_of,
    masked_value,
    member_spec,
)
from ._changes import CompiledRequest, NetworkRows, Topology
from ._requests import TopologyRequest, request_of


def _request_network(spec: KernelSpec, declarations: Declarations) -> tuple[object, str]:
    """Return the network a request of a spec changes and the rows the spec runs on."""
    if isinstance(spec.pairs, EdgeSearch):
        nodes = declarations.blocks[cast('str', spec.roles['source'])]
        return group_space(nodes), 'edge'
    if spec.group in declarations.capacity:
        return group_space(declarations.capacity[spec.group]), 'node'
    if spec.rows == 1:
        spaces = {id(block.node.space): block.node.space for block in declarations.capacity.values()}
        if len(spaces) == 1:
            return next(iter(spaces.values())), 'single'
    error_msg = (
        f'{spec.owner} requests topology changes, but it runs on neither the members of a variable group on the '
        'nodes or edges of a network nor once for the one network whose groups are variable. Give the groups of the '
        'network `variable=True`.'
    )
    raise ValidationError(error_msg)


def _node_block(space: object, declarations: Declarations) -> Block:
    """Return the first variable group on the nodes of a network."""
    for block in declarations.capacity.values():
        if getattr(block.node.component, 'on', None) == 'sites' and block.node.space is space:
            return block
    error_msg = 'A topology request changes a network without a variable group on its nodes. Add one.'
    raise ValidationError(error_msg)


def _toward_specs(specs: list[tuple[KernelSpec, list[BaseLaw]]], declarations: Declarations) -> list[Any]:
    """Return the specs of the nodes of every request with a weight, each with the law of the weight to lower."""
    towards: list[tuple[KernelSpec, list[BaseLaw]]] = []
    for spec, laws in specs:
        for law in laws:
            request = request_of(law)
            if request is None or request.toward is None:
                continue
            space, _ = _request_network(spec, declarations)
            nodes = _node_block(space, declarations)
            node_spec = member_spec(nodes, declarations.time)
            towards.append((node_spec, [UpdateLaw(_Weight(request), SR(request.toward(node_spec.namespace)))]))
    return towards


def _networks(declarations: Declarations) -> dict[int, NetworkRows]:
    """Return the variable groups of every network whose groups are variable, by the identity of the space."""
    spaces = {id(block.node.space): block.node.space for block in declarations.capacity.values()}
    networks = {}
    for key, space in spaces.items():
        nodes = tuple(
            capacity_rows(block, declarations)
            for block in declarations.capacity.values()
            if block.node.space is space and getattr(block.node.component, 'on', None) == 'sites'
        )
        if not nodes:
            continue
        sizes = {group.active.rows for group in nodes}
        if len(sizes) > 1:
            error_msg = (
                f'The variable groups on the nodes of one network, {", ".join(group.path for group in nodes)}, have '
                'different capacities. Give them one capacity.'
            )
            raise ValidationError(error_msg)
        edges = {
            layer: capacity_rows(block, declarations) for layer, block in declarations.changing_edges(space).items()
        }
        directed = {layer: cast('NetworkSpace', space).resolve_graph(layer).is_directed() for layer in edges}
        networks[key] = NetworkRows(nodes, edges, directed)
    return networks


def _request_kernel(spec: KernelSpec, value: object, declarations: Declarations) -> Kernel:
    """Return the kernel of the probability or the weight of a request for every row of its spec."""
    states, parameters = Layout(declarations.states), Layout(declarations.parameters)
    expression = SR(value)
    return Kernel(
        spec.owner,
        declarations.time,
        spec.references,
        spec.rows,
        states,
        parameters,
        spec.pairs,
        [Output(None, expression)],
    )


def _compiled_request(
    spec: KernelSpec,
    request: TopologyRequest,
    networks: dict[int, NetworkRows],
    weights: dict[int, Kernel],
    declarations: Declarations,
) -> CompiledRequest:
    """Return a request of a spec with its network, its rows, and the kernel of its weights."""
    space, rows = _request_network(spec, declarations)
    network = networks.get(id(space))
    if network is None:
        error_msg = f'{spec.owner} requests {request!r} on a network without variable groups on its nodes.'
        raise ValidationError(error_msg)
    if (request.on_edges or request.edges) and request.layer not in network.edges:
        error_msg = (
            f'{spec.owner} requests {request!r}, which changes the edges of the layer {request.layer!r}, but the '
            'layer has no variable edge group. Add `GroupComponent(..., on="edges", variable=True)` on the layer.'
        )
        raise ValidationError(error_msg)
    needs = {'remove-edge': 'edge', 'rewire': 'edge', 'add-edge': 'node', 'remove-node': 'node'}.get(request.kind)
    if needs is not None and needs != rows:
        error_msg = f'{spec.owner} requests {request!r}, which runs for every {needs}, from laws of every {rows}.'
        raise ValidationError(error_msg)
    return CompiledRequest(request, spec.owner, network, rows, toward=weights.get(id(request)))


def _request_owner(spec: KernelSpec, law: BaseLaw, agents: set[str]) -> str | None:
    """Return the group a topology request of a law runs for, or `None` for a law that targets states."""
    del agents
    if request_of(law) is None:
        return None
    return spec.group or next(iter(spec.groups), spec.owner)


def _split_requests(spec: KernelSpec, laws: list[BaseLaw]) -> tuple[list[BaseLaw], list[BaseLaw]]:
    """Return the laws of a spec without its update laws that request topology changes, and those update laws."""
    updates: list[BaseLaw] = [law for law in laws if isinstance(law, UpdateLaw) and request_of(law) is not None]
    kept = [law for law in laws if not any(law is item for item in updates)]
    return kept, updates


def _masked_requests(spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations) -> list[BaseLaw]:
    """Return update laws that request topology changes, with probabilities that vanish on free rows."""
    mask = mask_of(spec, declarations)
    if mask is None:
        return laws
    return [UpdateLaw(law.target, masked_value(law.value, mask)) if isinstance(law, UpdateLaw) else law for law in laws]


def _set_request_jumps_apart(
    specs: Laws,
) -> tuple[Laws, dict[int, tuple[KernelSpec, TopologyRequest]]]:
    """Return laws whose jumps that request topology changes change no state, with the request of each such jump."""
    requests: dict[int, tuple[KernelSpec, TopologyRequest]] = {}
    result: Laws = []
    for spec, laws in specs:
        kept: list[BaseLaw] = []
        for law in laws:
            request = request_of(law) if isinstance(law, JumpLaw) else None
            if not isinstance(law, JumpLaw) or request is None:
                kept.append(law)
                continue
            jump = JumpLaw(law.rate, {})
            requests[id(jump)] = (spec, request)
            kept.append(jump)
        result.append((spec, kept))
    return result, requests


def _is_weight(law: BaseLaw) -> bool:
    """Check whether a law sets the weight of the candidate nodes of a request."""
    return isinstance(getattr(law, 'target', None), _Weight)


def _topology(
    lowered: Lowered,
    apart: Laws,
    declarations: Declarations,
    reductions: ReductionPass,
    jumps: dict[int, tuple[KernelSpec, TopologyRequest]],
) -> Topology | None:
    """Return the topology pass of the variable groups of networks and the requests of laws, or `None` without them."""
    networks = _networks(declarations)
    if not networks:
        return None
    towards = [(spec, laws) for spec, laws in apart if all(_is_weight(law) for law in laws)]
    updates = [(spec, laws) for spec, laws in apart if not all(_is_weight(law) for law in laws)]
    weights = {
        id(law.target.request): _request_kernel(spec, law.value, declarations)
        for spec, laws in towards
        for law in laws
        if isinstance(law, UpdateLaw)
    }
    requests = []
    for spec, laws in updates:
        for law in laws:
            request = cast('TopologyRequest', request_of(law))
            item = _compiled_request(spec, request, networks, weights, declarations)
            requests.append(replace(item, condition=_request_kernel(spec, cast('UpdateLaw', law).value, declarations)))
    channels = {}
    index = 0
    for kernel in lowered.kernels[JumpLaw]:
        for output in kernel.outputs:
            _, law = lowered.laws[id(output)]
            found = jumps.get(id(law))
            if found is not None:
                channels[index] = _compiled_request(*found, networks, weights, declarations)
            index += 1
    return Topology(tuple(networks.values()), tuple(requests), channels, reductions)


@dataclass(frozen=True, eq=False)
class _Weight:
    """Target of the weight of the candidate nodes of a topology request, lowered with the reductions of laws."""

    request: TopologyRequest


class SplitRequestsPass(BasePass):
    """Pass that sets the update laws that request topology changes apart from the laws lowered into kernels."""

    reads = ('collected', 'apart')
    writes = ('collected', 'apart')

    def run(self, compilation: Compilation) -> None:
        """Set the update laws that request topology changes apart.

        Args:
            compilation:
                The compilation state.
        """
        collected: BlockLaws = []
        apart = cast('Laws', compilation.apart)
        for spec, laws, block in cast('BlockLaws', compilation.collected):
            kept, updates = _split_requests(spec, laws)
            collected.append((spec, kept, block))
            if updates:
                apart.append((spec, updates))
        compilation.collected = collected


class MaskRequestsPass(BasePass):
    """Pass that makes the requests of update laws vanish on the free rows of capacity layouts."""

    reads = ('drawn_apart', 'layout')
    writes = ('drawn_apart',)

    def run(self, compilation: Compilation) -> None:
        """Mask the requests of the update laws set apart.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.drawn_apart = [
            (spec, _masked_requests(spec, laws, declarations)) for spec, laws in cast('Laws', compilation.drawn_apart)
        ]


class TowardsPass(BasePass):
    """Pass that sets apart the weight of the candidate nodes of every request, read through reductions like laws."""

    reads = ('drawn_apart', 'drawn', 'layout')
    writes = ('drawn_apart',)

    def run(self, compilation: Compilation) -> None:
        """Set the weights of the requests apart.

        Args:
            compilation:
                The compilation state.
        """
        apart = cast('Laws', compilation.drawn_apart)
        towards = _toward_specs([*cast('Laws', compilation.drawn), *apart], cast('Declarations', compilation.layout))
        compilation.drawn_apart = [*apart, *towards]


class RequestJumpsPass(BasePass):
    """Pass that turns the jumps that request topology changes into jumps of no state change, noting the requests."""

    reads = ('reduced',)
    writes = ('reduced', 'request_jumps')

    def run(self, compilation: Compilation) -> None:
        """Note the requests of the jumps.

        Args:
            compilation:
                The compilation state.
        """
        compilation.reduced, compilation.facts['request_jumps'] = _set_request_jumps_apart(
            cast('Laws', compilation.reduced),
        )


class TopologyFunctionsPass(BasePass):
    """Pass that compiles the topology changes, leaving out the compiled stepper loops when laws request them."""

    reads = ('functions', 'kernels', 'reduced_apart', 'layout', 'reductions', 'request_jumps')
    writes = ('functions', 'topology')

    def run(self, compilation: Compilation) -> None:
        """Compile the topology changes of the networks.

        Args:
            compilation:
                The compilation state.
        """
        topology = _topology(
            cast('Lowered', compilation.kernels),
            cast('Laws', compilation.reduced_apart),
            cast('Declarations', compilation.layout),
            cast('ReductionPass', compilation.reductions),
            compilation.facts['request_jumps'],
        )
        if topology is not None and (topology.requests or topology.channels):
            cast('dict[str, Any]', compilation.functions)['accelerated'] = None
        compilation.facts['topology'] = topology


class TopologyProblemPass(BasePass):
    """Pass that adds the topology changes to the problem."""

    reads = ('problem', 'topology')
    writes = ('problem',)

    def run(self, compilation: Compilation) -> None:
        """Add the topology changes to the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.problem = replace(cast('Problem', compilation.problem), topology=compilation.facts['topology'])
