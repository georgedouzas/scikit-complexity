"""
# COMETS

A colony on a plate spreads because its biomass grows and diffuses. COMETS puts one metabolic network on every box of a
lattice and alternates two steps: each box solves its network and grows, and then biomass and glucose diffuse. This
example runs the colony expansion of Harcombe et al. (2014) on their lattice of 50 by 50 boxes of 0.02 cm, with their
diffusion coefficients, inoculum, and step of 0.1 h, for the first 14 h.
"""

# %%
# The model is created by name from the catalog, which needs the extra `metabolism`. The primary view is the image of
# the growth flux of every box, beside the images of the fields.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import IMEXEulerSolver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'comets'
SHAPE = (50, 50)
STEP = 0.1
INTERVAL = 4
INOCULUM_PER_LEVEL = 3.0

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=INTERVAL)
simulation = Simulation(create(MODEL, shape=SHAPE), solver=IMEXEulerSolver(dt=STEP), dt=STEP, observers=[observer])
trajectory = simulation.prepare().run(until=14.0).trajectory_

# %%
# The speed of the colony
# -----------------------
# Harcombe et al. count the boxes above 1e-7 g of biomass, a third of the inoculum of 3e-7 g. The radius of the area of
# those boxes grows at a nearly constant speed after a transient. The glucose diffuses about 17000 times faster than
# the biomass, so the glucose falls under the colony and also far beyond its edge.

times = trajectory.times
biomass = np.reshape(trajectory.values('biomass.x'), (times.size, *SHAPE))
glucose = np.reshape(trajectory.values('glucose.c'), (times.size, *SHAPE))
level = biomass[0].max() / INOCULUM_PER_LEVEL
radii = np.sqrt(np.count_nonzero(biomass > level, axis=(1, 2)) / np.pi)
late = times > times[-1] / 3
speed = np.polyfit(times[late], radii[late], 1)[0]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(times, radii)
axes[0].set_xlabel('time (h)')
axes[0].set_ylabel('radius (boxes)')
axes[1].imshow(biomass[-1], cmap='viridis')
axes[1].set_title('biomass (gDW/L)')
axes[2].imshow(glucose[-1], cmap='viridis')
axes[2].set_title('glucose (mM)')
fig.suptitle(f'The colony spreads at {speed:.2f} boxes/h after the transient')
fig.tight_layout()
plt.show()
