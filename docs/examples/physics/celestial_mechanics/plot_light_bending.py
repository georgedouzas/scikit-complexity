"""
# Light bending

A light ray that grazes a mass bends towards it by twice the angle Newton's gravity gives a particle at the speed of
light, the effect the solar eclipse of 1919 measured.
"""

# %%
# The ray follows a null geodesic of the Schwarzschild metric of the mass `M = 1`, in units with `G = c = 1`. It starts
# 2000 units behind the mass, offset by the impact parameter `b = 20`, and runs along the axis. The default blueprint
# draws the ray in the plane of its path.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.celestial_mechanics import LightBendingSystem

SYSTEM = 'light_bending'

system = LightBendingSystem(impact_parameter=20.0, distance=2000.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
simulation = simulate(system, solver=solver, observers=[observer])

# %%
# Deflection against the impact parameter
# ---------------------------------------
# The direction of the ray after it passes the mass gives its deflection, close to `4 M / b` with a correction of order
# `(M / b)²`. Rays of several impact parameters show the deflection falling as `1 / b`.

impacts = np.array([10.0, 20.0, 40.0, 80.0])
deflections = []
for impact in impacts:
    ray = LightBendingSystem(impact_parameter=impact, distance=2000.0)
    trajectory = simulate(ray, until=4000.0, dt=4.0, solver=solver)
    x = np.asarray(trajectory.get_values('photon.x'))[-1]
    u = np.asarray(trajectory.get_values('photon.x_d1'))[-1]
    vx = u[1] * np.cos(x[3]) - x[1] * np.sin(x[3]) * u[3]
    vy = u[1] * np.sin(x[3]) + x[1] * np.cos(x[3]) * u[3]
    deflections.append(-np.arctan2(vy, vx))
x = np.asarray(simulation.get_values('photon.x'))
fig, (path, angle) = plt.subplots(1, 2, figsize=(10, 4))
path.plot(x[:, 1] * np.cos(x[:, 3]), x[:, 1] * np.sin(x[:, 3]))
path.plot(0, 0, '*', color='black', markersize=10)
path.set_xlim(-300, 300)
path.set_ylim(-60, 60)
path.set_title('The ray bends towards the mass')
angle.loglog(impacts, deflections, 'o', label='simulated')
angle.loglog(impacts, 4 / impacts, '--', color='black', label='4 M / b')
angle.set_xlabel('impact parameter b')
angle.set_ylabel('deflection (rad)')
angle.legend()
fig.tight_layout()
plt.show()
