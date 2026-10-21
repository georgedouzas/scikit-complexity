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

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'winfree'

system = create(MODEL, n_oscillators=50, K=0.5, spread=0.1, seed=0)
blueprint = default_blueprint(system, primary='observables')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=0.25, observers=[observer]).run(until=50)

# %%
# Incoherence, locking, and death
# -------------------------------
# Over a sweep of the coupling, the spread of the mean frequencies of the oscillators should vanish once they lock,
# and at strong coupling the mean frequency itself should drop to zero as the oscillators stop, Ariaratnam and
# Strogatz's oscillator death.

couplings = np.linspace(0.0, 1.5, 31)
means, spreads = [], []
for coupling in couplings:
    run = Simulation(create(MODEL, n_oscillators=50, K=coupling, spread=0.1, seed=0), dt=0.25).run(until=200)
    times, phases = np.asarray(run.trajectory_.times), np.asarray(run.trajectory_['oscillators.theta'])
    late = times >= 100
    frequencies = (phases[late][-1] - phases[late][0]) / (times[late][-1] - times[late][0])
    means.append(frequencies.mean())
    spreads.append(frequencies.std())
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(couplings, means, 'o-', label='mean frequency')
ax.plot(couplings, spreads, 's-', label='spread of the frequencies')
ax.set_xlabel('coupling K')
ax.set_ylabel('frequency')
ax.set_title('Winfree model: locking, then oscillator death')
ax.legend()
plt.show()
