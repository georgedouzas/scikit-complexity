"""
# Exponential growth

A population with a constant per-capita growth rate grows without limit: it doubles in the same time whatever its
size, so its curve bends ever more steeply upwards.
"""

# %%
# The system is built from its class with a growth rate of 0.2 and run for 20 time units, about six doubling times.
# The observer saves every frame with the default blueprint, whose primary view is the time series of the population.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import ExponentialGrowthSystem

SYSTEM = 'exponential_growth'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ExponentialGrowthSystem(n=10.0, r=0.2), observers=[observer])

# %%
# A straight line on a logarithmic scale
# --------------------------------------
# The law `dn/dt = r n` has the solution `n(t) = n0 exp(r t)`, so `log n` grows linearly with slope `r` and the
# population doubles every `log 2 / r`. The simulated population should lie on the straight line.

times, population = trajectory.times, trajectory.get_values('population.x')
r, n0 = 0.2, 10.0
slope = np.polyfit(times, np.log(population), 1)[0]
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(times, population, label='simulated')
ax.semilogy(times, n0 * np.exp(r * times), '--', color='black', label='n0 exp(r t)')
for doubling in np.arange(0, times[-1], np.log(2) / r):
    ax.axvline(doubling, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel('time')
ax.set_ylabel('population (log scale)')
ax.set_title(f'Fitted rate {slope:.3f}, system rate {r}; dotted lines every doubling time')
ax.legend()
plt.show()
