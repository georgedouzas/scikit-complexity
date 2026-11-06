"""
# Kerr frame dragging

A particle dropped without angular momentum towards a spinning mass does not fall straight in: the spin drags space
around with it, and the particle turns faster the deeper it falls.
"""

# %%
# The particle follows a timelike geodesic of the Kerr metric of the mass `M = 1` and the spin `a = 0.9`, in units with
# `G = c = 1`, starting at rest at the radius 10 in the equatorial plane. The default blueprint draws its path.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'kerr_frame_dragging'

system = create_system(SYSTEM, radius=10.0, spin=0.9)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
simulation = Simulation(system, dt=0.1, solver=solver, observers=[observer]).prepare().run(until=33.0)

# %%
# Angular velocity of the frames
# ------------------------------
# A particle without angular momentum turns with the frames at its radius, at the angular velocity
# `dφ/dt = 2 M a / (r³ + a² r + 2 M a²)` that the term `g_tφ` of the metric sets.

x = np.asarray(simulation.trajectory_.values('particle.x'))
u = np.asarray(simulation.trajectory_.values('particle.x_d1'))
radius, angle = x[:, 1], x[:, 3]
fig, (path, rate) = plt.subplots(1, 2, figsize=(10, 4))
path.plot(radius * np.cos(angle), radius * np.sin(angle))
path.add_patch(plt.Circle((0, 0), 1 + np.sqrt(1 - 0.81), color='black'))
path.set_aspect('equal')
path.set_title('The fall winds around the spin axis')
rate.plot(radius, u[:, 3] / u[:, 0], label='simulated')
rate.plot(radius, 1.8 / (radius**3 + 0.81 * radius + 1.62), '--', color='black', label='frame dragging')
rate.set_xlabel('radius r')
rate.set_ylabel('dφ/dt')
rate.legend()
fig.tight_layout()
plt.show()
