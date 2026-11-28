"""Compile the changes of the shape of networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
from dataclasses import replace

from sage.all__sagemath_symbolics import SR

from ....components import EntrySpec
from ....components.laws import BaseLaw, JumpLaw, TopologyRequestSpec, UpdateLaw
from ....components.spaces import NetworkSpace
from ....enums import VariableKind
from ....errors import PipelineError, ValidationError
from ....types import Array
from ..._utils import declarations_of, declared_variables, default_of
from ...problems import LayoutSpec
from ...problems._edge_search import EdgeSearch
from ...problems._kernel_spec import KernelSpec
from ...problems._law_kernel import LawKernel, _Output
from ...problems._lowered_kernels_spec import LoweredKernelsSpec
from ...problems._neighborhood_kernel import _ReductionSchedule
from ...problems._topology_kernel import TopologyKernel, _CapacityRows, _CompiledRequest, _NetworkRows
from .._base import BasePass
from .._compilation import ACTIVE, ENDS, MEMBERS, Compilation, Laws, _Block, _Declarations
from ._layout import LayoutPass
from ._specs import SpecsPass


@dataclasses.dataclass(frozen=True, eq=False)
class _Weight:
    """Target of the weight or the score of the candidate nodes of a topology request.

    It is lowered like laws.
    """

    request: TopologyRequestSpec
    score: bool = False


class TopologyFunctionsPass(BasePass):
    """A pass that compiles the changes of the shape of networks.

    It compiles every request for a new or removed node or edge into the functions that apply it during a run, such as
    the newcomer of a growing network who links to popular nodes.

    Read more in the user guide, under Extend the library.

    See Also:
        TopologyProblemPass: Adds the changes to the problem.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.mathematics.networks import BarabasiAlbertSystem
        >>> compilation = PassCompiler().prepare(BarabasiAlbertSystem()).run().compilation_
        >>> [request.owner for request in compilation.topology.requests]
        ['growth']
    """

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
        topology = self._get_topology(
            Compilation._get_required(compilation.kernels, 'kernels'),
            Compilation._get_required(compilation.reduced_apart, 'reduced_apart'),
            Compilation._get_required(compilation.layout, 'layout'),
            Compilation._get_required(compilation.reductions, 'reductions'),
            jumps,
        )
        if topology is not None and (topology.requests or topology.channels):
            compilation.functions = replace(functions, accelerated=None)
        compilation.topology = topology

    @staticmethod
    def _request_kernel(spec: KernelSpec, value: object, declarations: _Declarations) -> LawKernel:
        """Return the kernel of the probability or the weight of a request.

        It is evaluated for every row of its spec.
        """
        states, parameters = LayoutSpec(tuple(declarations.states)), LayoutSpec(tuple(declarations.parameters))
        expression = SR(value)
        return LawKernel(
            spec.owner,
            declarations.time,
            spec.references,
            spec.rows,
            states,
            layout_parameters=parameters,
            pairs=spec.pairs,
            outputs=[_Output(None, expression)],
        ).prepare()

    @staticmethod
    def _get_compiled_request(
        spec: KernelSpec,
        request: TopologyRequestSpec,
        networks: dict[int, _NetworkRows],
        weights: dict[tuple[int, bool], LawKernel],
        declarations: _Declarations,
    ) -> _CompiledRequest:
        """Return a request of a spec with its network, rows, and kernels.

        The kernels are those of its weights and its score.
        """
        space, rows = TopologyFunctionsPass._request_network(spec, declarations)
        network = networks.get(id(space))
        if network is None:
            error_msg = f'{spec.owner} requests {request!r} on a network without variable groups on its nodes.'
            raise ValidationError(error_msg)
        if TopologyKernel._changes_edges(request) and request.layer not in network.edges:
            error_msg = (
                f'{spec.owner} requests {request!r}, which changes the edges of the layer {request.layer!r}, but the '
                'layer has no variable edge group. Add `Element(..., on="edges")` on the layer.'
            )
            raise ValidationError(error_msg)
        if request.hyperedge is not None and network.hyperedges is None:
            error_msg = (
                f'{spec.owner} requests {request!r}, which forms a hyperedge, but the network has no variable group '
                'on its hyperedges. Add `Element(..., on="hyperedges")`.'
            )
            raise ValidationError(error_msg)
        if request.hyperedge is not None and network.hyperedges is not None:
            TopologyFunctionsPass._check_hyperedge_size(spec, request, network.hyperedges, declarations)
        needs = {
            'remove-edge': 'edge',
            'rewire': 'edge',
            'add-edge': 'node',
            'remove-node': 'node',
            'replace-node': 'node',
            'add-hyperedge': 'node',
            'remove-hyperedge': 'hyperedge',
            'close-triangle': 'node',
            'contract-edge': 'node',
            'duplicate-node': 'node',
        }.get(request.kind)
        once = request.kind == 'add-edge' and rows == 'single'
        if needs is not None and needs != rows and not once:
            error_msg = f'{spec.owner} requests {request!r}, which runs for every {needs}, from laws of every {rows}.'
            raise ValidationError(error_msg)
        if request.kind == 'remove-hyperedge' and network.hyperedges is None:
            error_msg = f'{spec.owner} removes hyperedges of a network without a variable group on its hyperedges.'
            raise ValidationError(error_msg)
        return _CompiledRequest(
            request,
            spec.owner,
            network,
            rows,
            toward=weights.get((id(request), False)),
            score=weights.get((id(request), True)),
        )

    @staticmethod
    def _check_hyperedge_size(
        spec: KernelSpec,
        request: TopologyRequestSpec,
        hyperedges: _CapacityRows,
        declarations: _Declarations,
    ) -> None:
        """Raise an error for a request that forms hyperedges of the wrong size.

        The rows of the network hold hyperedges of another size. A variable group on hyperedges with a rank holds
        hyperedges of every size up to it, and one without a rank holds hyperedges of the size of its initial
        hyperedges.

        Args:
            spec:
                The spec of the requesting laws.

            request:
                The request, which forms a hyperedge.

            hyperedges:
                The rows of the variable group on the hyperedges of the network.

            declarations:
                The declarations of the compilation.

        Raises:
            ValidationError:
                The hyperedge is larger than the rank, or of another size than the hyperedges of a group without one.
        """
        width = hyperedges.members.columns if hyperedges.members is not None else 0
        size = int(request.hyperedge or 0)
        ranked = getattr(declarations.blocks[hyperedges.path].node.component, 'rank', None) is not None
        if size == width or (ranked and size < width):
            return
        limit = f'at most {width} members, the rank of the group' if ranked else f'size {width}'
        error_msg = (
            f'{spec.owner} requests {request!r}, which forms hyperedges of size {size}, but the hyperedges of the '
            f'network have {limit}. Give the request such a size, or give the group a larger rank.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _is_weight(law: BaseLaw) -> bool:
        """Check whether a law sets the weight of the candidate nodes of a request."""
        return isinstance(next(iter(law._get_targets()), None), _Weight)

    @staticmethod
    def _get_topology(
        lowered: LoweredKernelsSpec,
        apart: Laws,
        declarations: _Declarations,
        reductions: _ReductionSchedule,
        jumps: dict[int, tuple[KernelSpec, TopologyRequestSpec]],
    ) -> TopologyKernel | None:
        """Return the topology pass of the network groups and law requests, or `None`.

        The groups are the variable groups of networks, and the result is `None` if there are none.
        """
        networks = TopologyFunctionsPass._get_networks(declarations)
        if not networks:
            return None
        towards = [(spec, laws) for spec, laws in apart if all(TopologyFunctionsPass._is_weight(law) for law in laws)]
        updates = [
            (spec, laws) for spec, laws in apart if not all(TopologyFunctionsPass._is_weight(law) for law in laws)
        ]
        weights = {
            (id(target.request), target.score): TopologyFunctionsPass._request_kernel(spec, law.value, declarations)
            for spec, laws in towards
            for law in laws
            if isinstance(law, UpdateLaw) and isinstance(target := law._get_targets()[0], _Weight)
        }
        requests = []
        for spec, laws in updates:
            for law in laws:
                request = TopologyKernel._request_of(law)
                if request is None or not isinstance(law, UpdateLaw):
                    error_msg = f'A law of {spec.owner} set apart as a change of a network requests no change.'
                    raise PipelineError(error_msg)
                item = TopologyFunctionsPass._get_compiled_request(spec, request, networks, weights, declarations)
                requests.append(
                    replace(item, condition=TopologyFunctionsPass._request_kernel(spec, law.value, declarations)),
                )
        channels = {}
        index = 0
        for kernel in lowered.kernels[JumpLaw]:
            for output in kernel.outputs:
                _, law = lowered.laws[id(output)]
                found = jumps.get(id(law))
                if found is not None and found[1].one:
                    error_msg = (
                        f'{found[0].owner} fires {found[1]!r} at a rate for one drawn member. A jump channel already '
                        'belongs to one member, so give `one=True` to a request with a probability per step.'
                    )
                    raise ValidationError(error_msg)
                if found is not None:
                    channels[index] = TopologyFunctionsPass._get_compiled_request(
                        *found,
                        networks,
                        weights,
                        declarations,
                    )
                index += 1
        return TopologyKernel(tuple(networks.values()), tuple(requests), channels, reductions).prepare()

    @staticmethod
    def _get_capacity_rows(block: _Block) -> _CapacityRows:
        """Return the rows of a variable group with the values of one new member.

        Args:
            block:
                The block of the variable group.

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
            if name in {ACTIVE, *ENDS, MEMBERS}:
                continue
            variables[name] = (entry, name in states)
            default = default_of(element, name) if name in declared_variables(element) else 0.0
            default = 0.0 if default is dataclasses.MISSING else default
            defaults[name] = LayoutPass._coerce_value(entry.name, default, entry.member_shape, entry.domain, None)
        source, target, members = (block.entries.get(name) for name in (*ENDS, MEMBERS))
        return _CapacityRows(node.path, variables, defaults, block.entries[ACTIVE], source, target, members)

    @staticmethod
    def _get_networks(declarations: _Declarations) -> dict[int, _NetworkRows]:
        """Return the variable groups of every network whose groups are variable.

        They are keyed by the identity of the space.
        """
        spaces = {id(block.node.space): block.node.space for block in declarations.capacity.values()}
        networks = {}
        for key, space in spaces.items():
            nodes = tuple(
                TopologyFunctionsPass._get_capacity_rows(block)
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
                layer: TopologyFunctionsPass._get_capacity_rows(block)
                for layer, block in declarations.get_changing_edges(space).items()
            }
            if not isinstance(space, NetworkSpace):
                error_msg = f'The variable groups {", ".join(group.path for group in nodes)} are not on a network.'
                raise ValidationError(error_msg)
            directed = {layer: space.graphs_[layer].is_directed() for layer in edges}
            hyperedges = next(
                (
                    TopologyFunctionsPass._get_capacity_rows(block)
                    for block in declarations.capacity.values()
                    if block.node.space is space and getattr(block.node.component, 'on', None) == 'hyperedges'
                ),
                None,
            )
            networks[key] = _NetworkRows(nodes, edges, directed, hyperedges)
        return networks

    @staticmethod
    def _request_network(spec: KernelSpec, declarations: _Declarations) -> tuple[object, str]:
        """Return the network a request of a spec changes and the rows the spec runs on."""
        if isinstance(spec.pairs, EdgeSearch):
            source = spec.roles[declarations.blocks[spec.owner].node.as_interaction._get_role('source')]
            if not isinstance(source, str):
                error_msg = f'The source role of {spec.owner} binds several groups, not the nodes of one network.'
                raise ValidationError(error_msg)
            nodes = declarations.blocks[source]
            return SpecsPass._get_group_space(nodes), 'edge'
        if spec.group in declarations.capacity:
            block = declarations.capacity[spec.group]
            on = getattr(block.node.component, 'on', None)
            return SpecsPass._get_group_space(block), 'hyperedge' if on == 'hyperedges' else 'node'
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
