"""Share the helper of the network system tests."""

import networkx as nx

from skcomplex.core.components.systems import BaseSystem as OldBaseSystem
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.systems import BaseSystem

from ..conftest import _run


def _final_graph(system: BaseSystem | OldBaseSystem, until: int, seed: int = 0) -> tuple[nx.Graph, Trajectory]:
    """Return the graph of the last frame of a run of a network whose topology changes."""
    trajectory = _run(system, until, dt=max(1, until // 10), random_state=seed)
    return trajectory.get_network(-1).get_graph(), trajectory
