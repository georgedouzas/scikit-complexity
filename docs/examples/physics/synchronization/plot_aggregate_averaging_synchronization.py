"""
# Aggregate averaging synchronization

Two chaotic logistic maps that each update from a weighted average of both states fall into perfect step: after a
short transient their values coincide forever, although each keeps wandering chaotically.
"""

# %%
# The maps start at 0.3 and 0.6 with `r = 4`, fully chaotic, and the coupling `κ = 0.4`. The default blueprint shows
# both values. After a few steps the two curves lie on top of each other, so only one colour remains visible.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'aggregate_averaging_synchronization'

system = create_system(SYSTEM, x1=0.3, x2=0.6, r=4.0, kappa=0.4)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).prepare().run(until=50)

# %%
# Window of synchronization
# -------------------------
# A small difference between the maps is multiplied at every step by `(1 - 2 κ) f'(x)`, and the logistic map at `r = 4`
# has the Lyapunov exponent `ln 2`. Synchrony is stable where `ln |1 - 2 κ| + ln 2 < 0`, the window
# `1/4 < κ < 3/4`, shaded. The difference left after 200 steps should vanish inside the window and stay of order one
# outside it.

couplings = np.linspace(0.0, 1.0, 41)
differences = []
for coupling in couplings:
    run = Simulation(create_system(SYSTEM, x1=0.3, x2=0.6, kappa=coupling), dt=1).prepare().run(until=200)
    gap = np.abs(np.ravel(run.trajectory_.values('first.x')) - np.ravel(run.trajectory_.values('second.x')))
    differences.append(gap[-50:].mean())
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(couplings, np.maximum(differences, 1e-17), 'o-')
ax.axvspan(0.25, 0.75, color='grey', alpha=0.2, label='stable synchrony')
ax.set_xlabel('coupling κ')
ax.set_ylabel('mean |x1 - x2| over the last 50 steps')
ax.set_title('Synchronization of two chaotic logistic maps')
ax.legend()
plt.show()

# %%
# Delay-coupled Bernoulli maps
# ----------------------------
# Two Bernoulli shift maps `f(a) = 1.5 a mod 1` coupled through the value of the other map `T` steps ago synchronize
# only for couplings inside a window that the delay shrinks, where every root of `z^(T+1) - 1.5 (1 - κ) z^T + 1.5 κ`
# lies in the unit disk. The difference should vanish inside the window of each delay and stay of order one outside.

fig, ax = plt.subplots(figsize=(6, 4))
couplings = np.linspace(0.05, 0.95, 19)
for delay in (0, 1, 2):
    gaps = []
    for coupling in couplings:
        system = create_system(SYSTEM, kind='bernoulli', r=1.5, kappa=coupling, delay=delay)
        run = Simulation(system, dt=1).prepare().run(until=300)
        gap = np.abs(np.ravel(run.trajectory_.values('first.x')) - np.ravel(run.trajectory_.values('second.x')))
        gaps.append(np.minimum(gap, 1 - gap)[-50:].max())
    ax.semilogy(couplings, np.maximum(gaps, 1e-17), 'o-', markersize=3, label=f'delay T = {delay}')
ax.set_xlabel('coupling κ')
ax.set_ylabel('largest |x1 - x2| over the last 50 steps')
ax.set_title('Delay-coupled Bernoulli maps')
ax.legend()
plt.show()
