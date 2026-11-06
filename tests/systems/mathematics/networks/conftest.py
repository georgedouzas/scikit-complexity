"""Share the helper of the network system tests."""

import networkx as nx

from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation, Trajectory


def _final_graph(system: System, until: int, seed: int = 0) -> tuple[nx.Graph, Trajectory]:
    """Return the graph of the last frame of a run of a network whose topology changes."""
    trajectory = Simulation(system, dt=max(1, until // 10), random_state=seed).prepare().run(until=until).trajectory_
    return trajectory.network(-1).graph(), trajectory
