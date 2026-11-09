"""
# Adaptive SIS with rewiring

Susceptible nodes cut their edges to infected neighbours and rewire them to other susceptible nodes. The network
reorganises around the epidemic: infected nodes lose their contacts, susceptible nodes cluster, and the prevalence
stays far below its value on the same network without rewiring.
"""

# %%
# The system is built from its class on an Erdős-Rényi network of 200 nodes with mean degree 6, run by the Gillespie
# solver. The observer saves every frame with the default blueprint, whose primary view is the network drawn by its
# force layout, with the infected nodes bright, and its edges as they are rewired.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.epidemiology import AdaptiveSISSystem

SYSTEM = 'adaptive_sis'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(AdaptiveSISSystem(w=1.0), observers=[observer], random_state=1)

# %%
# Prevalence with and without rewiring
# ------------------------------------
# Rewiring lowers the rate of new infections, so the prevalence should fall with the rewiring rate, while the number
# of edges stays the same. The share of edges between two susceptible nodes should rise above its value without
# rewiring, the clustering of the susceptible nodes.

fig, ax = plt.subplots(figsize=(8, 4))
for w in (0.0, 1.0, 4.0):
    trajectory = simulate(AdaptiveSISSystem(w=w), random_state=1)
    snapshot = trajectory.get_network(-1)
    infected = np.nan_to_num(trajectory.get_values('hosts.i')[-1])
    healthy = np.mean((infected[snapshot.sources] == 0) & (infected[snapshot.targets] == 0))
    label = f'w = {w}: S-S edges {healthy:.0%}'
    ax.plot(trajectory.times, np.nanmean(trajectory.get_values('hosts.i'), axis=1), label=label)
ax.set_xlabel('time')
ax.set_ylabel('prevalence')
ax.set_title('Rewiring away from infected neighbours suppresses the epidemic')
ax.legend()
plt.show()
