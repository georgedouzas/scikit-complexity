"""Changing topologies: requests of laws that add and remove the nodes and edges of networks with variable groups."""

from .._formalism import Formalism, register
from ._changes import CompiledRequest, NetworkRows, Topology
from ._checks import _check_requests
from ._passes import (
    MaskRequestsPass,
    RequestJumpsPass,
    SplitRequestsPass,
    TopologyFunctionsPass,
    TopologyProblemPass,
    TowardsPass,
    _request_owner,
)
from ._requests import TopologyRequest, add_edge, add_node, remove_edge, remove_node, request_of, rewire

__all__: list[str] = [
    'CompiledRequest',
    'MaskRequestsPass',
    'NetworkRows',
    'RequestJumpsPass',
    'SplitRequestsPass',
    'Topology',
    'TopologyFunctionsPass',
    'TopologyProblemPass',
    'TopologyRequest',
    'TowardsPass',
    'add_edge',
    'add_node',
    'remove_edge',
    'remove_node',
    'request_of',
    'rewire',
]

register(
    Formalism(
        'topology',
        passes=(
            SplitRequestsPass(),
            MaskRequestsPass(),
            TowardsPass(),
            RequestJumpsPass(),
            TopologyFunctionsPass(),
            TopologyProblemPass(),
        ),
        checks=(_check_requests,),
        owners=(_request_owner,),
    ),
)
