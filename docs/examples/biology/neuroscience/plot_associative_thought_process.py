"""
# Associative thought process

A winners-take-all network stores memories as cliques of sites. Slow reservoirs make every memory a transient
attractor: once the reservoirs of the active clique run low, the activity hops to an associated clique that shares
sites with it, and the network runs through its memories one after the other.
"""

# %%
# The system is built from its class with the seven sites and six cliques of Gros (2015, figure 10.12), started in the
# clique (0, 1). The observer saves every frame with the default blueprint, whose primary view is the time series of
# the activities.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import AssociativeThoughtProcessSystem

SYSTEM = 'associative_thought_process'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=3)
trajectory = simulate(AssociativeThoughtProcessSystem(), observers=[observer])

# %%
# Transient attractors
# --------------------
# Every lasting coalition of active sites should be one of the stored cliques, (0, 1), (0, 6), (3, 6), (1, 2, 3),
# (4, 5, 6), or (1, 2, 4, 5), and the reservoirs of a clique should fall while it is active and recover after.

activities, reservoirs = trajectory.get_values('sites.x'), trajectory.get_values('sites.phi')
fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
top.imshow(activities.T, aspect='auto', origin='lower', cmap='Greys', extent=(0, trajectory.times[-1], -0.5, 6.5))
top.set_ylabel('site')
top.set_title('Activities: the memories become active one after the other')
for site in range(activities.shape[1]):
    bottom.plot(trajectory.times, reservoirs[:, site], label=f'site {site}')
bottom.set_xlabel('time')
bottom.set_ylabel('reservoir phi')
bottom.legend(ncol=7, fontsize=7)
winners = [tuple(np.flatnonzero(row > 0.85)) for row in activities]
lasting = sorted({winner for winner in winners if len(winner) > 1 and winners.count(winner) > 20})
bottom.set_title(f'Lasting coalitions {lasting}', fontsize=9)
fig.tight_layout()
plt.show()
