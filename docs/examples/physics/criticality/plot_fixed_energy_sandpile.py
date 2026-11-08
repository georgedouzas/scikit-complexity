"""
# Fixed-energy sandpile

A sandpile without drive and without loss: grains topple on a periodic lattice, so their number never changes. Below a
critical density the topplings stop in an absorbing configuration, and above it the pile stays active forever, an
absorbing phase transition whose order parameter is the density of active sites.
"""

# %%
# The pile of 32 x 32 sites starts with 2.3 grains per site scattered at random, above the critical density. The
# default blueprint shows the heights, which keep toppling.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.criticality import FixedEnergySandpileSystem

SYSTEM = 'fixed_energy_sandpile'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=5)
trajectory = simulate(FixedEnergySandpileSystem(), until=500, observers=[observer], random_state=0)

# %%
# Absorbing phase transition
# --------------------------
# The density of active sites after 500 steps should be zero below a critical density near 2.125, dashed, and grow
# continuously above it.

densities = np.linspace(1.9, 2.6, 15)
activities = []
for density in densities:
    run = simulate(FixedEnergySandpileSystem(density=density), until=500, random_state=0)
    activities.append(np.mean(run.values('activity')[-100:]))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(densities, activities, 'o-')
ax.axvline(2.125, linestyle='--', color='black')
ax.set_xlabel('density of grains')
ax.set_ylabel('density of active sites')
ax.set_title('Absorbing phase transition of the fixed-energy sandpile')
plt.show()
