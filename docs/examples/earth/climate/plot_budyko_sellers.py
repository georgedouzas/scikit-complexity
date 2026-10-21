"""
# Budyko-Sellers climate

Started uniformly warm, the hemisphere cools fastest at the pole, where the sun is weakest, and settles into a climate
of warm tropics and a polar ice cap. Under the same sun, a hemisphere started cold freezes over entirely: a snowball
Earth.
"""

# %%
# The model is created by name from the catalog with 30 bands in the sine of latitude, all at 15 °C, and present
# insolation. The default blueprint puts the history of the temperature first, with the equator on the left and the pole
# on the right, over the first 16 time units, while the pole cools from 15 °C to below -10 °C and the equator warms.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'budyko_sellers'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.25, observers=[observer]).run(until=16)

# %%
# Two climates under one sun
# --------------------------
# The steady temperature should fall from the equator to the pole, with an ice cap where it is below -10 °C. A run
# started at -40 °C should instead stay below -10 °C everywhere, since the high albedo of ice reflects enough sunlight
# to keep it frozen.

x = (np.arange(30) + 0.5) / 30
warm = fields_of(simulation.trajectory_, 'bands.T')[-1]
cold = fields_of(Simulation(create(MODEL, T=-40.0), dt=2.0).run(until=100).trajectory_, 'bands.T')[-1]
latitude = np.degrees(np.arcsin(x))
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(latitude, warm, 'o-', markersize=3, color='C3', label='started at 15 °C')
ax.plot(latitude, cold, 'o-', markersize=3, color='C0', label='started at -40 °C')
ax.axhline(-10.0, color='black', linestyle='--', linewidth=0.8, label='ice below -10 °C')
ax.set_xlabel('latitude (degrees)')
ax.set_ylabel('steady temperature (°C)')
ax.set_title('Ice cap climate and snowball Earth')
ax.legend()
plt.show()
