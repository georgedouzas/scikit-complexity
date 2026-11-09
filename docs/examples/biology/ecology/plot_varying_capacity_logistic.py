"""
# Logistic growth with a seasonal capacity

A population that grows logistically under a carrying capacity that rises and falls with the seasons forgets where
it started and settles on a cycle of one season. The cycle lags behind the capacity and swings less than it.
"""

# %%
# The system is built from its class and run from two populations, one far below and one far above the mean capacity
# of 100, for eight seasons. The observer saves every frame with the default blueprint, whose primary view is the time
# series of the population.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import VaryingCapacityLogisticSystem

SYSTEM = 'varying_capacity_logistic'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(VaryingCapacityLogisticSystem(n=5.0), observers=[observer], until=8)

# %%
# A unique periodic solution
# --------------------------
# The capacity is `K (1 + amplitude sin(2 π t / period))`. Started from 5 or from 180, the population should converge
# to the same cycle, which peaks after the capacity and stays inside its range.

low = trajectory
high = simulate(VaryingCapacityLogisticSystem(n=180.0), until=8)
capacity = 100.0 * (1 + 0.5 * np.sin(2 * np.pi * low.times))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(low.times, capacity, color='grey', linestyle=':', label='carrying capacity')
ax.plot(low.times, low.values('population.x'), label='started at 5')
ax.plot(high.times, high.values('population.x'), '--', label='started at 180')
ax.set_xlabel('time (seasons)')
ax.set_ylabel('population')
ax.set_title('Both start points converge to one lagging cycle')
ax.legend()
plt.show()
