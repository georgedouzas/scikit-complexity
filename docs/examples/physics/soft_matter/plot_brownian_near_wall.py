"""
# Brownian motion near a wall

A colloid settles under gravity towards the bottom wall of its cell, but the fluid squeezed between the particle and
the wall slows its motion, so its diffusion coefficient vanishes as it approaches the wall. Released high above the
wall, the particle sinks and then wanders through a layer a few sedimentation lengths thick, its path smoothest when
it comes close to the wall. The view shows the height of the particle above the wall.
"""

# %%
# The system is built from its class with the diffusion coefficient `D0 = 1` far from the wall, the
# hindrance length `a = 1`, the settling velocity `0.2`, and the anti-Itô convention, released at height 20. The
# Euler-Maruyama stepper takes steps of 0.01 and the observer saves every frame of 1 with the default blueprint,
# whose primary view is the time series of the height.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.soft_matter import BrownianNearWallSystem

SYSTEM = 'brownian_near_wall'

system = BrownianNearWallSystem(z=20.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = EulerMaruyamaSolver(dt=0.01)
trajectory = simulate(system, until=250, solver=solver, dt=1.0, observers=[observer], random_state=0)

# %%
# Barometric equilibrium
# ----------------------
# The drift is the settling velocity times the same mobility that sets the diffusion, so in the anti-Itô convention
# the particle reaches thermal equilibrium in gravity whatever the hindrance: the heights follow the barometric law
# `p(z) = (v / D0) exp(-v z / D0)` with the sedimentation length `D0 / v = 5`. A long run started at the wall should
# fill this exponential profile.

run = simulate(BrownianNearWallSystem(z=1.0), until=15000, solver=EulerMaruyamaSolver(dt=0.02), dt=0.5, random_state=1)
heights = run.values('particle.z')[200:]
histogram = distribution(heights, kind='histogram', bins=30)
edges, density = histogram.edges, histogram.density
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
