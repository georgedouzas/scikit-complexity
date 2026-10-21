"""
# Stommel's thermohaline circulation

The overturning of the ocean has two stable modes: a strong circulation driven by the temperature difference between
the equator and the pole, and a weak reversed one driven by salinity. Started with no salinity difference, the boxes
settle in the thermal mode.
"""

# %%
# The model is created by name from the catalog with the forcings `eta1 = 3`, `eta2 = 1.2`, and `eta3 = 0.3`, starting
# at `T = 3` and `S = 0`. The default blueprint shows the temperature and salinity differences in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stommel'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=50)

# %%
# Two basins of circulation
# -------------------------
# The flow is proportional to the density difference `T - S`. Runs from a grid of initial differences should end either
# in the thermal mode, with `T > S`, or in the haline mode, with `S > T`, so the phase plane splits into two basins.

fig, ax = plt.subplots(figsize=(6, 5))
for T0 in np.linspace(0.1, 2.9, 8):
    for S0 in np.linspace(0.1, 2.9, 8):
        run = Simulation(create(MODEL, T=T0, S=S0), dt=0.1).run(until=50).trajectory_
        T, S = run['ocean.T'], run['ocean.S']
        ax.plot(T, S, color='C3' if T[-1] > S[-1] else 'C0', linewidth=0.7)
        ax.plot(T[-1], S[-1], 'o', color='black', markersize=6)
ax.plot([0, 3], [0, 3], color='gray', linestyle='--', label='no flow, T = S')
ax.set_xlabel('temperature difference T')
ax.set_ylabel('salinity difference S')
ax.set_title('Thermal mode (red) and haline mode (blue)')
ax.legend()
plt.show()
