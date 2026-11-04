"""
# COMETS

A colony on a plate spreads because its biomass diffuses into the glucose around it. COMETS puts one metabolic network
on every site of a lattice, solves it only where there is biomass, and lets biomass and glucose diffuse, so the edge of
the colony moves at a nearly constant speed.
"""

# %%
# The model is created by name from the catalog, which needs the extra `metabolism`. The primary view is the image of
# the growth flux of every site, beside the images of the fields.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'comets'
SHAPE = (25, 25)
LEVEL = 0.01

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(create(MODEL, shape=SHAPE), solver=RK4Solver(dt=0.05), dt=0.1, observers=[observer])
trajectory = simulation.check().prepare().run(until=5.0).trajectory_

# %%
# The speed of the colony
# -----------------------
# The radius of the area above a level of biomass should grow at a nearly constant speed after a transient, and the
# glucose under the colony should fall.

times = trajectory.times
biomass = np.reshape(trajectory.values('biomass.x'), (times.size, *SHAPE))
glucose = np.reshape(trajectory.values('glucose.c'), (times.size, *SHAPE))
radii = np.sqrt(np.count_nonzero(biomass > LEVEL, axis=(1, 2)) / np.pi)
late = times > times[-1] / 3
speed = np.polyfit(times[late], radii[late], 1)[0]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(times, radii)
axes[0].set_xlabel('time (h)')
axes[0].set_ylabel('radius (sites)')
axes[1].imshow(biomass[-1], cmap='viridis')
axes[1].set_title('biomass')
axes[2].imshow(glucose[-1], cmap='viridis')
axes[2].set_title('glucose')
fig.suptitle(f'The colony spreads at {speed:.2f} sites/h after the transient')
fig.tight_layout()
plt.show()
