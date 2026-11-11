"""Test the `_topology_request_spec` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import TopologyRequestSpec


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
    ],
)
def test_topology_request_spec_checks_its_kind_end_and_edges(spec, message):
    """Test that a topology request needs a known kind and kept end, whole edges, and a hyperedge of a new node."""
    with pytest.raises(ValidationError, match=message):
        spec._check()
