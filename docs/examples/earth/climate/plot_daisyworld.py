"""
# Daisyworld

White daisies cool their planet and black daisies warm it. The two populations settle at areas that hold the planet
near the temperature where daisies grow best, and they keep holding it there over a wide range of luminosities of the
star.
"""

# %%
# The model is created by name from the catalog with a fifth of the planet covered by each kind of daisy, at the
# luminosity `L = 1`. The default blueprint shows the areas of white and black daisies in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'daisyworld'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# Temperature regulation across luminosities
# ------------------------------------------
# A bare planet of albedo 0.5 has the effective temperature `(S L (1 - 0.5) / sigma)^(1/4)`, which grows steadily with
# the luminosity. With daisies, the steady temperature should stay close to the optimum of 22.5 °C over the range of
# luminosities where daisies survive, as black daisies give way to white ones.

S, sigma = 917.0, 5.67e-8
luminosities = np.linspace(0.6, 1.6, 41)
temperatures, whites, blacks = [], [], []
for luminosity in luminosities:
    run = Simulation(create(MODEL, L=luminosity, white=0.05, black=0.05), dt=1.0).run(until=200).trajectory_
    temperatures.append(run['temperature'][-1] - 273.15)
    whites.append(run['planet.white'][-1])
    blacks.append(run['planet.black'][-1])
bare = (S * luminosities * 0.5 / sigma) ** 0.25 - 273.15
fig, (climate, areas) = plt.subplots(1, 2, figsize=(10, 4))
climate.plot(luminosities, bare, color='gray', linestyle='--', label='bare planet')
climate.plot(luminosities, temperatures, 'o-', markersize=3, label='with daisies')
climate.axhline(22.5, color='black', linewidth=0.8, label='optimum for daisies')
climate.set_xlabel('luminosity L')
climate.set_ylabel('temperature (°C)')
climate.set_title('Daisies regulate the temperature')
climate.legend()
areas.plot(luminosities, whites, color='C7', label='white daisies')
areas.plot(luminosities, blacks, color='black', label='black daisies')
areas.set_xlabel('luminosity L')
areas.set_ylabel('area covered')
areas.set_title('Steady areas of the daisies')
areas.legend()
fig.tight_layout()
plt.show()
