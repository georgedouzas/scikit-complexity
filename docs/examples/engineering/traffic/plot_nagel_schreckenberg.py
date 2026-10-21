"""
# Nagel-Schreckenberg traffic

Cars on a ring road accelerate up to a speed limit, keep their distance, and sometimes brake at random. At this density
the random braking sets off jams out of nowhere, which drift backwards against the traffic as cars leave their front
and join their back.
"""

# %%
# The model is created by name from the catalog on a ring of 1000 cells at the density 0.2, with the speed limit 5 and
# the braking probability 0.3, and runs for 200 steps. The default blueprint shows the road over time, where the jams
# appear as dense stripes that lean backwards.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'nagel_schreckenberg'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=1, observers=[observer]).run(until=200)

# %%
# The fundamental diagram
# -----------------------
# The flow is the density times the mean speed of the cars. At low densities every car drives near the speed limit less
# the random braking, so the flow should grow as `(vmax - p) rho`; above a critical density near 0.15 jams take over and
# the flow should fall as the road fills up.

densities = np.linspace(0.02, 0.8, 27)
flows = []
for density in densities:
    run = Simulation(create(MODEL, density=float(density), random_state=1), dt=1).run(until=200).trajectory_
    speeds = fields_of(run, 'road.velocity')[100:]
    flows.append(density * speeds[speeds >= 0].mean())
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(densities, flows, 'o-', markersize=3, label='simulation')
ax.plot(densities[:6], (5 - 0.3) * densities[:6], color='black', linestyle='--', label='free flow (vmax - p) rho')
ax.axvline(0.2, color='C3', linewidth=0.8, label='density of the run above')
ax.set_xlabel('density of cars')
ax.set_ylabel('flow (cars per step)')
ax.set_title('Free flow below and jams above a critical density')
ax.legend()
plt.show()
