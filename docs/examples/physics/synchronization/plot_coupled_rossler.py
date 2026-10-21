"""
# Coupled Rössler

Two chaotic Rössler oscillators with slightly different frequencies, weakly coupled, keep their chaotic amplitudes but
lock their phases: the phase difference stays bounded instead of drifting, chaotic phase synchronization.
"""

# %%
# The frequencies are `1 ± 0.02` and the coupling through the second variable is `k = 0.06`, above the threshold of
# phase synchronization. The default blueprint shows the three variables of both oscillators, whose peaks keep in step
# while their heights differ.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'coupled_rossler'

system = create(MODEL, n_oscillators=2, k=0.06, mismatch=0.02, seed=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=100)

# %%
# Phase difference
# ----------------
# The phase of each oscillator is the angle `atan2(x2, x1)` of its rotation in the plane of its first two variables.
# Without coupling the phase difference should drift by about `2 * 0.02` per unit time, and above the threshold it
# should stay bounded within a fraction of a turn, although the amplitudes stay chaotic and uncorrelated.

fig, ax = plt.subplots(figsize=(6, 4))
for coupling in [0.0, 0.02, 0.06]:
    run = Simulation(create(MODEL, n_oscillators=2, k=coupling, mismatch=0.02, seed=0), dt=0.1).run(until=500)
    phases = [
        np.unwrap(np.arctan2(np.ravel(run.trajectory_[f'{name}.x2']), np.ravel(run.trajectory_[f'{name}.x1'])))
        for name in ('oscillator_0', 'oscillator_1')
    ]
    ax.plot(run.trajectory_.times, (phases[1] - phases[0]) / (2 * np.pi), label=f'k = {coupling}')
ax.set_xlabel('time')
ax.set_ylabel('phase difference (turns)')
ax.set_title('Phase locking of two chaotic oscillators')
ax.legend()
plt.show()
