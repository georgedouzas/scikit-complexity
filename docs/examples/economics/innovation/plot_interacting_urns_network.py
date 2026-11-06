"""
# Interacting urns on a network

Iacopini, Di Bona, Ubaldi, Loreto, and Latora's discovery processes on a social network: every member of Zachary's
karate club explores with an urn with triggering, drawing from its own urn and from the colours of the urns of its
friends. A draw of a colour new to the member adds new colours to its urn, the adjacent possible, and its friends can
draw them too. Members in central places of the network discover faster than members at its edge.
"""

# %%
# Every urn starts with four balls of its own colours and draws with the reinforcement `rho = 6` and the triggering
# `nu = 3`, the parameters of Iacopini et al. (2020). The urns are agents on the nodes, and the balls of every urn are
# agents of a variable group on its node, one agent per colour with its multiplicity.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'interacting_urns_network'
STEPS = 150

graph = nx.karate_club_graph()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = (
    Simulation(create_system(SYSTEM, graph=graph, rho=6, nu=3), observers=[observer], random_state=0)
    .prepare()
    .run(until=STEPS)
)

# %%
# Heaps' law and Eq. (4)
# ----------------------
# The novelties `D_i(t)` of every member grow as Heaps' law. Eq. (4) of the article, the expected growth of the
# novelties from the sizes of the social urns, should rank the members as the simulation does, with the instructor,
# node 0, and the administrator, node 33, ahead of the members with few friends.


def heaps(adjacency, rho, nu, steps):
    """Return the novelties of every member at every step by Eq. (4) of Iacopini et al. (2020)."""
    identity = np.eye(len(adjacency))
    novelties = [np.zeros(len(adjacency))]
    for step in range(steps):
        current = novelties[-1]
        numerator = (nu + 1) * (adjacency + identity).sum(axis=1) + (nu * identity + (nu + 1) * adjacency) @ current
        balls = ((adjacency + identity) * (nu + 1 + (nu + 1) * current)[None, :]).sum(axis=1)
        novelties.append(current + numerator / (rho * step + balls))
    return np.asarray(novelties)


trajectory = simulation.trajectory_
frames = range(len(trajectory.times))
novelties = np.asarray(
    [
        trajectory.agents('urns', frame).values['novelties'][np.argsort(trajectory.agents('urns', frame).locations)]
        for frame in frames
    ],
)
expected = heaps(nx.to_numpy_array(graph), 6, 3, STEPS)
degrees = np.asarray([degree for _, degree in graph.degree()])
fig, (growth, ranking) = plt.subplots(1, 2, figsize=(10, 4))
for node, color in ((0, 'C0'), (33, 'C1'), (int(np.argmin(degrees)), 'C2')):
    growth.loglog(trajectory.times[1:], novelties[1:, node], 'o', ms=2, color=color, label=f'node {node}')
    growth.loglog(np.arange(1, STEPS + 1), expected[1:, node], '-', color=color)
growth.set_xlabel('step t')
growth.set_ylabel('novelties D_i(t)')
growth.set_title('Simulation (dots) and Eq. (4) (lines)')
growth.legend()
ranking.scatter(expected[-1], novelties[-1], c=degrees, cmap='viridis')
ranking.set_xlabel('D_i by Eq. (4)')
ranking.set_ylabel('simulated D_i')
ranking.set_title(f'Novelties after {STEPS} steps, colored by degree')
fig.tight_layout()
plt.show()
