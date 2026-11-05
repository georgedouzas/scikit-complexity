"""Test the `_weighted_edge` module of `skcomplex.systems.biology.common.networks`."""

from skcomplex.core.specs import StateSpec
from skcomplex.systems.biology.common.networks import WeightedEdgeElement


def test_weighted_edge_holds_a_weight_and_a_mark():
    """Test that a weighted edge declares its weight and its mark as states, the mark set by default."""
    specs = WeightedEdgeElement._get_specs()
    assert all(isinstance(specs[name], StateSpec) for name in ('w', 'fresh'))
    assert WeightedEdgeElement('edge', w=0.5).fresh == 1.0
