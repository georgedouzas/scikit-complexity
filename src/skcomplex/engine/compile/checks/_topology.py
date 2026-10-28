"""Check that the topology requests of laws change only variable groups, with a variable edge group per layer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.elements import GroupElement
from ....core.errors import ValidationError
from ....core.spaces import NetworkSpace
from ....core.specs import TopologyRequestSpec
from .._census import Census, CensusRecord
from .._compilation import Compilation
from .._issues import Issue
from .._topology import Topology
from .._tree import Node


def _fixed_groups(
    request: TopologyRequestSpec,
    owner: object,
    groups: list[Node],
) -> tuple[list[str], bool]:
    """Return the fixed groups a request changes, and whether its layer lacks an edge group."""
    edges = [node for node in groups if node.component.on == 'edges' and node.component.layer == request.layer]
    sites = [node for node in groups if node.component.on == 'sites']
    changed = []
    if Topology._changes_edges(request):
        changed += edges
    if request.kind in {'add-node', 'remove-node'}:
        changed += sites
    if owner is not None:
        changed.append(owner)
    fixed = list(dict.fromkeys(node.path for node in changed if not node.component.variable))
    return fixed, Topology._changes_edges(request) and not edges


def _request_issues(records: list[CensusRecord], nodes: list[Node]) -> list[Issue]:
    """Return the topology requests that change fixed groups, or edges of a layer without a variable edge group."""
    by_path = {node.path: node for node in nodes}
    issues = []
    for record in records:
        try:
            request = Topology._request_of(record.law)
        except ValidationError as error:
            issues.append(Issue('value', record.path, str(error), 'Fire one request per jump, with increment 1.'))
            continue
        if request is None:
            continue
        space = record.space
        if not isinstance(space, NetworkSpace):
            spaces = {id(node.space): node.space for node in nodes if isinstance(node.space, NetworkSpace)}
            space = next(iter(spaces.values())) if len(spaces) == 1 else space
        groups = [node for node in nodes if isinstance(node.component, GroupElement) and node.space is space]
        owner = by_path.get(record.owner)
        owner = owner if owner is not None and isinstance(owner.component, GroupElement) else None
        fixed, missing = _fixed_groups(request, owner, [node for node in groups if node.component.on])
        if not fixed and not missing:
            continue
        listed = ', '.join(fixed) or 'none'
        where = f'the layer {request.layer!r}' if request.layer is not None else 'the network'
        message = (
            f'The {type(record.law).__name__} of {record.path} requests {request!r}, which changes the fixed '
            f'groups {listed}.'
        )
        if missing:
            message += f' {where[0].upper()}{where[1:]} has no group on its edges.'
        fix = (
            'Give the groups on the nodes and the edges of the network `variable=True`, with a group on the edges '
            'of every layer the request changes, `GroupElement(..., on="edges", variable=True)`.'
        )
        issues.append(Issue('group-fixed', record.path, message, fix))
    return issues


def request_issues(compilation: Compilation) -> list[Issue]:
    """Return the topology requests of a compilation that change fixed groups or a layer without an edge group."""
    return _request_issues(cast('Census', compilation.census).records, cast('list[Node]', compilation.nodes))
