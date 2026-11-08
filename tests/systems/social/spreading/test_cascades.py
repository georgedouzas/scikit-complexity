"""Test the `_cascades` module of `skcomplex.systems.social.spreading`."""

import networkx as nx

from skcomplex.systems.social.spreading import WattsCascadesSystem

from ..conftest import _run, _values


def test_watts_cascades_depend_on_threshold():
    """Test that one seed on a ring activates every node at threshold 0.5 and only itself at threshold 0.6."""
    spread = _run(WattsCascadesSystem(nx.cycle_graph(20), threshold=0.5, random_state=0), 30)
    assert _values(spread, 'nodes', 'active').sum() == 20
    stuck = _run(WattsCascadesSystem(nx.cycle_graph(20), threshold=0.6, random_state=0), 30)
    assert _values(stuck, 'nodes', 'active').sum() == 1
