"""Test the `_layer_spec` module of `skcomplex.core.specs`."""

import networkx as nx
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import LayerSpec


def test_layer_spec_needs_a_string_label():
    """Test that a layer is labelled by a string."""
    assert LayerSpec('work', nx.path_graph(3)).label == 'work'
    with pytest.raises(ValidationError, match='string'):
        LayerSpec(1, nx.path_graph(3))._check()
