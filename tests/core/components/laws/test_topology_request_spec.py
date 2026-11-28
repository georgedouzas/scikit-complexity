"""Test the `_topology_request_spec` module of `skcomplex.core.components.laws`."""

import pytest

from skcomplex.core.components.laws import TopologyRequestSpec
from skcomplex.core.errors import ValidationError


@pytest.mark.parametrize(
    ('spec', 'message'),
    [
        (TopologyRequestSpec('split'), 'Unknown topology request'),
        (TopologyRequestSpec('rewire', keep='middle'), 'source or the target'),
        (TopologyRequestSpec('add-node', edges=1.5), 'whole number'),
        (TopologyRequestSpec('add-node', edges=-1), 'whole number'),
        (TopologyRequestSpec('add-node', hyperedge=1), 'at least 2 members'),
        (TopologyRequestSpec('add-edge', hyperedge=3), 'forms no hyperedge'),
        (TopologyRequestSpec('add-edge', candidates=0), 'at least 1 candidates'),
        (TopologyRequestSpec('add-node', candidates=2), 'chooses among candidates'),
        (TopologyRequestSpec('add-edge', candidates=2, independent=True), 'chooses among candidates'),
        (TopologyRequestSpec('add-edge', score=3), 'function of the namespace'),
        (TopologyRequestSpec('duplicate-node', inherit=1.5), r'in \[0, 1\]'),
        (TopologyRequestSpec('duplicate-node', parent=-0.1), r'in \[0, 1\]'),
        (TopologyRequestSpec('add-node', further=1.5), 'whole number of further edges'),
        (TopologyRequestSpec('add-node', further=1, closing=1.5), r'in \[0, 1\]'),
        (TopologyRequestSpec('add-edge', further=1), 'adds further edges'),
        (TopologyRequestSpec('add-node', edges=0, further=1), 'adds further edges'),
        (TopologyRequestSpec('add-node', further=1, independent=True), 'adds further edges'),
        (TopologyRequestSpec('add-node', further=1, hyperedge=3), 'adds further edges'),
    ],
)
def test_topology_request_spec_checks_its_kind_end_and_edges(spec, message):
    """Test that a topology request needs a known kind and kept end, whole edges, and a hyperedge of a new node."""
    with pytest.raises(ValidationError, match=message):
        spec._check()
