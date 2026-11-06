"""
# Social dilemmas on a simplicial complex

Players of a growing network of triangles play one game along the links of a plain triangle and, in a 2-simplex, a
milder game against a member when the third member shares their strategy. Every player copies a random neighbour with
the Fermi probability of their payoffs, so the share of 2-simplices changes how much cooperation survives.
"""

# %%
# Five hundred players start half as cooperators on a network where half of the triangles are 2-simplices. The default
# blueprint shows the strategies on the network.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'simplicial_social_dilemmas'
ROUNDS = 100

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
system = create_system(SYSTEM)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=ROUNDS)

# %%
# Cooperation against the share of 2-simplices
# --------------------------------------------
# The share of 2-simplices sets how many triangles play the milder game, so the share of cooperators at the end of runs
# should rise with it.


def cooperators(simplices):
    """Return the share of cooperators at the end of a run."""
    run = Simulation(create_system(SYSTEM, simplices=simplices), random_state=0).prepare().run(until=ROUNDS)
    return np.asarray(run.trajectory_.values('players.strategy'))[-1].mean()


shares = np.linspace(0.0, 1.0, 6)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(shares, [cooperators(share) for share in shares], 'o-')
ax.set_xlabel('share of the triangles that are 2-simplices')
ax.set_ylabel(f'share of cooperators after {ROUNDS} rounds')
ax.set_title('Higher-order games change cooperation')
plt.show()
