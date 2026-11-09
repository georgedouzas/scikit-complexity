"""
# Duplication-deletion network

Every step a random node dies with its links, and a random parent is duplicated. The offspring inherits each link of
its parent with probability `inherit`, links to its parent with probability `parent`, and links to every other node
with a small probability `new`. Deaths and births balance, and the number of links settles at a steady value.
"""

# %%
# A hundred nodes start as a random graph of link probability 0.05, and every step removes one node and duplicates
# another, with `inherit = 0.5`, `parent = 0.5`, and `new = 0.01`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import DuplicationDeletionSystem

SYSTEM = 'duplication_deletion'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(DuplicationDeletionSystem(), until=4000, dt=20.0, observers=[observer], random_state=0)


def connectance(inherit, n_nodes=100, until=4000):
    """Return every 40th step and the connectance of the network then, the share of its pairs that are linked."""
    system = DuplicationDeletionSystem(n_nodes=n_nodes, inherit=inherit)
    run = simulate(system, until=until, random_state=0)
    steps = np.arange(0, until + 1, 40)
    links = np.array([len(run.get_network(int(step)).sources) for step in steps])
    return steps, links / (n_nodes * (n_nodes - 1) / 2)


# %%
# Steady connectance
# ------------------
# The connectance should relax to Eq. 8.84 of Jensen, `C = (new (N - 2) + parent) / (N - 1 + (new - inherit)
# (N - 2))`, from any starting network, at a level that rises with the fidelity `inherit` of the duplication.

fig, axis = plt.subplots(figsize=(6, 4))
for index, inherit in enumerate((0.3, 0.5, 0.7)):
    axis.plot(*connectance(inherit), color=f'C{index}', linewidth=0.8, label=f'inherit = {inherit}')
    axis.axhline((0.01 * 98 + 0.5) / (99 + (0.01 - inherit) * 98), color=f'C{index}', linestyle='--')
axis.set_xlabel('step')
axis.set_ylabel('connectance')
axis.set_title('Simulated connectance (lines) and Eq. 8.84 (dashed)')
axis.legend()
fig.tight_layout()
plt.show()
