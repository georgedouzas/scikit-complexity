"""
# Penaeid shrimp

One shrimp eats, digests, breathes, and spawns under a year of changing food and water temperature, while its
carapace grows along von Bertalanffy's curve towards its asymptotic length.
"""

# %%
# The system is created by name from the catalog and run for eight months of days. The observer saves every other
# frame with the default blueprint, whose primary view is the time series of the shrimp's stocks.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'penaeid_shrimp'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).prepare().run(until=240)

# %%
# Von Bertalanffy growth
# ----------------------
# The carapace length starts at zero and follows `dL/dt = K (L_inf - L)`, so it should lie on `L_inf (1 - exp(-K
# t))` with `L_inf = 29.9` mm and `K = 0.013` per day.

trajectory = simulation.trajectory_
times = trajectory.times
fig, (length, stocks) = plt.subplots(1, 2, figsize=(11, 4.5))
length.plot(times, trajectory.values('shrimp.length'), label='model')
length.plot(times, 29.9 * (1 - np.exp(-0.013 * times)), '--', color='black', label='von Bertalanffy')
length.set_xlabel('day')
length.set_ylabel('carapace length (mm)')
length.set_title('Length growth')
length.legend()
for state in ('gut', 'biomass', 'gonads'):
    stocks.plot(times, trajectory.values(f'shrimp.{state}'), label=state)
stocks.set_xlabel('day')
stocks.set_ylabel('dry weight (g)')
stocks.set_title('Stocks of the shrimp')
stocks.legend()
fig.tight_layout()
plt.show()
