"""Share the helper of the network model tests."""

import networkx as nx

from skcomplex.analysis import as_graph
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory


def _final_graph(system: System, until: int, seed: int = 0) -> tuple[nx.Graph, Trajectory]:
    """Return the graph of the last frame of a run of a network whose topology changes."""
    trajectory = Simulation(system, dt=max(1, until // 10), random_state=seed).run(until=until).trajectory_
    return as_graph(trajectory.network(-1)), trajectory
