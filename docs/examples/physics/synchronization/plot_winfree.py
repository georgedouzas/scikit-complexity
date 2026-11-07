"""
# Winfree

Oscillators that send pulses and respond to the pulses of all the others lock to one common frequency at moderate
coupling, and at strong coupling the response wins over the drive and every oscillator stops: oscillator death.
"""

# %%
# Fifty oscillators with frequencies spread evenly over `[0.9, 1.1]` start at random phases with the coupling
# `K = 0.5`. The blueprint puts the order parameter first, rising to a steady value as the phases lock, with the phases
# in a second view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.synchronization import WinfreeSystem

SYSTEM = 'winfree'

system = WinfreeSystem(n_oscillators=50, K=0.5, spread=0.1, seed=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='observables', interval=1)
trajectory = simulate(system, dt=0.25, observers=[observer])

# %%
# Incoherence, locking, and death
# -------------------------------
# Over a sweep of the coupling, the spread of the mean frequencies of the oscillators should vanish once they lock,
# and at strong coupling the mean frequency itself should drop to zero as the oscillators stop, Ariaratnam and
# Strogatz's oscillator death.

couplings = np.linspace(0.0, 1.5, 31)
means, spreads = [], []
for coupling in couplings:
    system_system = WinfreeSystem(n_oscillators=50, K=coupling, spread=0.1, seed=0)
    run = simulate(system_system, until=200, dt=0.25)
    times, phases = np.asarray(run.times), np.asarray(run.values('oscillators.theta'))
    late = times >= 100
    frequencies = (phases[late][-1] - phases[late][0]) / (times[late][-1] - times[late][0])
    means.append(frequencies.mean())
    spreads.append(frequencies.std())
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(couplings, means, 'o-', label='mean frequency')
ax.plot(couplings, spreads, 's-', label='spread of the frequencies')
ax.set_xlabel('coupling K')
ax.set_ylabel('frequency')
ax.set_title('Winfree system: locking, then oscillator death')
ax.legend()
plt.show()
