"""
# Zhang sandpile

Random packets of energy load the sites of a lattice, and a site whose energy passes the threshold gives all of it to
its four neighbours, which may pass it on in turn. Energy lost at the open edges balances the loading, and the pile
organizes itself into a critical state with avalanches of all sizes.
"""

# %%
# The pile of 32 x 32 sites starts from random energies. The default blueprint shows the energies between avalanches.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'zhang_sandpile'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run(until=10000)

# %%
# Avalanches and energies
# -----------------------
# In the critical state the sizes of the avalanches should spread over two decades with a heavy tail, and the energies
# of the sites, unlike their uniform start, should pile up near a few values, the quasi-units of the Zhang pile.

trajectory = simulation.trajectory_
sizes = np.array([event.size for event in trajectory.events if event.time > 3000])
energies = np.ravel(np.asarray(trajectory.values('pile.energy'))[-1])
bins = np.logspace(0, np.log10(sizes.max()), 20)
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.hist(sizes, bins=bins, density=True, log=True)
left.set_xscale('log')
left.set_xlabel('avalanche size')
left.set_ylabel('probability density')
right.hist(energies, bins=40)
right.set_xlabel('energy of a site')
right.set_ylabel('sites')
fig.tight_layout()
plt.show()
