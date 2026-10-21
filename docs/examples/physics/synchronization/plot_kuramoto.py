"""
# Kuramoto

Oscillators with different natural frequencies, each pulled towards the others, fall into step once the coupling is
strong enough: a synchronized cluster forms, and the order parameter `r` jumps from noise level to a steady high value.
"""

# %%
# Fifty oscillators with normally distributed frequencies of width `sigma = 1` start at random phases, with the coupling
# `K = 4`, well above the critical coupling. The blueprint puts the order parameter first, rising from about
# `1 / sqrt(N)` to near 1, with the phases in a second view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'kuramoto'

system = create(MODEL, n_oscillators=50, K=4.0, sigma=1.0, seed=0)
blueprint = default_blueprint(system, primary='observables')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=30)

# %%
# Onset of synchronization
# ------------------------
# For a normal density `g` of the frequencies, Kuramoto's theory puts the onset at `K_c = 2 / (π g(0))`, about 1.6 for
# `sigma = 1`. The time-averaged order parameter of 400 oscillators should stay at the finite-size level `~1 / sqrt(N)`
# below `K_c` and rise steeply above it.

couplings = np.linspace(0.0, 4.0, 17)
order = []
for coupling in couplings:
    run = Simulation(create(MODEL, n_oscillators=400, K=coupling, sigma=1.0, seed=1), dt=0.2).run(until=60)
    order.append(np.mean(np.asarray(run.trajectory_['order'])[-100:]))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(couplings, order, 'o-')
ax.axvline(2 * np.sqrt(2 * np.pi) / np.pi, linestyle='--', color='black', label='K_c = 2 / (π g(0))')
ax.set_xlabel('coupling K')
ax.set_ylabel('time-averaged order parameter r')
ax.set_title('Synchronization transition of 400 oscillators')
ax.legend()
plt.show()
