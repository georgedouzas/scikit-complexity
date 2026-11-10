"""
# Population and pollution

A growing population pollutes its environment, and the environment absorbs pollution ever more slowly as it
accumulates. Pollution then lowers births, food, and the quality of life, and raises deaths.
"""

# %%
# The run follows the pollution model of Bala et al. over 50 years, from a population of 100 million and a pollution
# of 10 million units. The default blueprint shows the stocks and the observables.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import PopulationPollutionSystem

SYSTEM = 'population_pollution'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(PopulationPollutionSystem(), until=50, observers=[observer], random_state=0)

# %%
# The four curves of Bala et al.
# ------------------------------
# As in Fig. 4.33 of Bala et al. (2017), the pollution should climb towards 90 million, the population double, the food
# per capita fall towards the subsistence level of 180, and the quality of life fall below 1.

times = np.asarray(trajectory.times)
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
for ax, name in zip(axes.ravel(), ['pollution', 'population', 'food_per_capita', 'quality_of_life'], strict=True):
    ax.plot(times, np.ravel(trajectory.get_values(f'region.{name}')))
    ax.set_title(name.replace('_', ' '))
for ax in axes[1]:
    ax.set_xlabel('years')
plt.show()
