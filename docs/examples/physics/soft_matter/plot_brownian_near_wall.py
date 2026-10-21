"""
# Brownian motion near a wall

A colloid settles under gravity towards the bottom wall of its cell, but the fluid squeezed between the particle and
the wall slows its motion, so its diffusion coefficient vanishes as it approaches the wall. Released high above the
wall, the particle sinks and then wanders through a layer a few sedimentation lengths thick, its path smoothest when
it comes close to the wall. The view shows the height of the particle above the wall.
"""

# %%
# The model is created by name from the catalog with the diffusion coefficient `D0 = 1` far from the wall, the
# hindrance length `a = 1`, the settling velocity `0.2`, and the anti-Itô convention, released at height 20. The
# Euler-Maruyama stepper takes steps of 0.01 and the observer saves every frame of 1 with the default blueprint,
# whose primary view is the time series of the height.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'brownian_near_wall'

system = create(MODEL, z=20.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.01)
simulation = Simulation(system, solver=solver, dt=1.0, observers=[observer], random_state=0).run(until=250)

# %%
# Barometric equilibrium
# ----------------------
# The drift is the settling velocity times the same mobility that sets the diffusion, so in the anti-Itô convention
# the particle reaches thermal equilibrium in gravity whatever the hindrance: the heights follow the barometric law
# `p(z) = (v / D0) exp(-v z / D0)` with the sedimentation length `D0 / v = 5`. A long run started at the wall should
# fill this exponential profile.

run = Simulation(create(MODEL, z=1.0), solver=EulerMaruyama(dt=0.02), dt=0.5, random_state=1).run(until=15000)
heights = run.trajectory_['particle.z'][200:]
edges, density = histogram(heights, bins=30)
grid = np.linspace(0, edges[-1], 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.stairs(density, edges, fill=True, alpha=0.5, label='simulation')
ax.plot(grid, 0.2 * np.exp(-0.2 * grid), color='black', label='barometric law, D0 / v = 5')
ax.set_xlabel('height z above the wall')
ax.set_ylabel('probability density')
ax.set_title('Hindered diffusion still reaches the barometric equilibrium')
ax.legend()
fig.tight_layout()
plt.show()
