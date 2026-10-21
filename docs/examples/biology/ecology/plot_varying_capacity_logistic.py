"""
# Logistic growth with a seasonal capacity

A population that grows logistically under a carrying capacity that rises and falls with the seasons forgets where
it started and settles on a cycle of one season. The cycle lags behind the capacity and swings less than it.
"""

# %%
# The model is created by name from the catalog and run from two populations, one far below and one far above the
# mean capacity of 100, for eight seasons. The observer saves every frame with the default blueprint, whose
# primary view is the time series of the population.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'varying_capacity_logistic'

system = create(MODEL, n=5.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=8)

# %%
# A unique periodic solution
# --------------------------
# The capacity is `K (1 + amplitude sin(2 π t / period))`. Started from 5 or from 180, the population should converge
# to the same cycle, which peaks after the capacity and stays inside its range.

low = simulation.trajectory_
high = Simulation(create(MODEL, n=180.0), dt=0.05).run(until=8).trajectory_
capacity = 100.0 * (1 + 0.5 * np.sin(2 * np.pi * low.times))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(low.times, capacity, color='grey', linestyle=':', label='carrying capacity')
ax.plot(low.times, low['population.n'], label='started at 5')
ax.plot(high.times, high['population.n'], '--', label='started at 180')
ax.set_xlabel('time (seasons)')
ax.set_ylabel('population')
ax.set_title('Both start points converge to one lagging cycle')
ax.legend()
plt.show()
