"""
# Soil temperature maps

The surface temperature of the soil `x` jumps chaotically from day to day, as a logistic map with `r = 3.8`, while the
deeper soil `y` averages it into small, damped fluctuations.
"""

# %%
# The system is created by name from the catalog with `r = 3.8` and the coupling `epsilon = 0.1`, and iterated for 200
# steps. The default blueprint shows both temperatures in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import OrbitDiagramExperiment
from skcomplex.systems import create_system, describe

SYSTEM = 'soil_temperature'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(system, dt=1, observers=[observer]).prepare().run(until=200)

# %%
# Route to chaos
# --------------
# As for the logistic map, the surface temperature should settle on the fixed point
# `x* = (r + epsilon^2 / (1 - epsilon) - 1) / r` for small `r`, then double its period repeatedly and turn chaotic as
# `r` grows towards 4, with the coupled run above in the chaotic range.

epsilon = 0.1
diagram = (
    OrbitDiagramExperiment(
        describe(SYSTEM).factory,
        'r',
        np.linspace(2.5, 3.95, 300),
        'soil.x',
        transient=300,
        keep=80,
    )
    .prepare()
    .run()
)
rates = np.linspace(2.5, 3.0, 50)
fig, ax = plt.subplots(figsize=(7, 5))
for rate, points in zip(diagram.values, diagram.points_, strict=True):
    ax.plot(np.full(len(points), rate), points, ',', color='C0', alpha=0.5)
ax.plot(rates, (rates + epsilon**2 / (1 - epsilon) - 1) / rates, color='black', label='fixed point x*')
ax.axvline(3.8, color='C3', linestyle='--', label='r of the run above')
ax.set_xlabel('r')
ax.set_ylabel('surface temperature x')
ax.set_title('Orbit diagram of the surface temperature')
ax.legend()
plt.show()
