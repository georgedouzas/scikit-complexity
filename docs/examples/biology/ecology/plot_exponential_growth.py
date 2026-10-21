"""
# Exponential growth

A population with a constant per-capita growth rate grows without limit: it doubles in the same time whatever its
size, so its curve bends ever more steeply upwards.
"""

# %%
# The model is created by name from the catalog with a growth rate of 0.2 and run for 20 time units, about six
# doubling times. The observer saves every frame with the default blueprint, whose primary view is the time series of
# the population.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'exponential_growth'

system = create(MODEL, n=10.0, r=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# A straight line on a logarithmic scale
# --------------------------------------
# The law `dn/dt = r n` has the solution `n(t) = n0 exp(r t)`, so `log n` grows linearly with slope `r` and the
# population doubles every `log 2 / r`. The simulated population should lie on the straight line.

times, population = simulation.trajectory_.times, simulation.trajectory_['population.n']
r, n0 = 0.2, 10.0
slope = np.polyfit(times, np.log(population), 1)[0]
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(times, population, label='simulated')
ax.semilogy(times, n0 * np.exp(r * times), '--', color='black', label='n0 exp(r t)')
for doubling in np.arange(0, times[-1], np.log(2) / r):
    ax.axvline(doubling, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel('time')
ax.set_ylabel('population (log scale)')
ax.set_title(f'Fitted rate {slope:.3f}, model rate {r}; dotted lines every doubling time')
ax.legend()
plt.show()
