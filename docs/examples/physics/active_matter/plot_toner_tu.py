"""
# Toner-Tu flock

The Toner-Tu equations describe a polar flock as a fluid: a density field carried by a velocity field that grows
spontaneously where the density exceeds a critical value, saturates at a fixed speed, and aligns with its
neighbours by viscosity and self-advection. Started from a disordered velocity field, patches of aligned motion grow
and merge until the whole fluid moves in one direction, while the density gathers into large, smooth patches. The view
shows the density of a 64 x 64 periodic lattice.
"""

# %%
# The model is created by name from the catalog with mean density `rho0 = 1` above the critical density
# `rho_c = 0.5`. The Euler-Maruyama stepper takes steps of 0.02 and the observer saves every frame of 0.25 with the
# default blueprint, whose primary view is the image of the density.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'toner_tu'

system = create(MODEL, shape=(64, 64), random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.02)
simulation = Simulation(system, solver=solver, dt=0.25, observers=[observer], random_state=0).run(until=30)

# %%
# The flocking transition
# -----------------------
# A uniform flock moves with the speed `sqrt(a (rho0 - rho_c) / beta)` above the critical density and is at rest
# below it. The mean velocity of the whole lattice, which is zero in a disordered field, should after a long run
# follow this curve against the mean density on a 16 x 16 lattice, with a polar order close to one above `rho_c`.

densities = np.linspace(0.2, 2.0, 10)
speeds = []
for density in densities:
    system_d = create(MODEL, shape=(16, 16), rho0=density, random_state=1)
    run = Simulation(system_d, solver=solver, dt=1.0, random_state=1).run(until=60)
    velocity = np.stack([run.trajectory_['flock.vx'][-10:], run.trajectory_['flock.vy'][-10:]], axis=-1)
    speeds.append(np.linalg.norm(velocity.mean(axis=1), axis=-1).mean())
grid = np.linspace(0.2, 2.0, 300)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(densities, speeds, 'o', label='speed of the mean velocity, simulation')
ax.plot(grid, np.sqrt(np.clip(grid - 0.5, 0, None)), color='black', label='sqrt(a (rho0 - rho_c) / beta)')
ax.axvline(0.5, color='gray', linestyle='--', label='rho_c')
ax.set_xlabel('mean density rho0')
ax.set_ylabel('flock speed')
ax.set_title('The fluid flocks above the critical density')
ax.legend()
fig.tight_layout()
plt.show()
