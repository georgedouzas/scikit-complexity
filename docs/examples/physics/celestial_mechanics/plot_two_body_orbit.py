"""
# Two-body orbit

A light planet around a heavy star traces a closed ellipse with the star at one focus, racing through its closest
approach and crawling at its farthest point, so that the line to the star sweeps equal areas in equal times.
"""

# %%
# The planet starts at its closest approach, 10 units from the star, with the speed of an orbit of eccentricity 0.6.
# Its period is `2π sqrt(a^3 / (G M))`, about 24.8 for the semi-major axis `a = 25`, and the run covers one period.
# The default blueprint shows the bodies in the plane, with the path of the planet.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'two_body_orbit'

system = create(MODEL, eccentricity=0.6, mass=1000.0, radius=10.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.05, solver='rk4', observers=[observer]).run(until=24.8)

# %%
# Kepler's first and second laws
# ------------------------------
# The orbit should lie on the ellipse `r(θ) = p / (1 + e cos θ)` with `p = 10 (1 + e)`, and the areal velocity
# `|x v_y - y v_x| / 2` should stay constant along the orbit.

relative = np.asarray(simulation.trajectory_['planet.x']) - np.asarray(simulation.trajectory_['star.x'])
velocity = np.asarray(simulation.trajectory_['planet.v']) - np.asarray(simulation.trajectory_['star.v'])
areal = np.abs(relative[:, 0] * velocity[:, 1] - relative[:, 1] * velocity[:, 0]) / 2
angles = np.linspace(0, 2 * np.pi, 400)
radius = 10 * 1.6 / (1 + 0.6 * np.cos(angles))
fig, (orbit, area) = plt.subplots(1, 2, figsize=(10, 4))
orbit.plot(radius * np.cos(angles), radius * np.sin(angles), color='black', linewidth=3, alpha=0.3, label='ellipse')
orbit.plot(relative[:, 0], relative[:, 1], label='simulated')
orbit.plot(0, 0, '*', color='orange', markersize=12, label='star')
orbit.set_aspect('equal')
orbit.set_xlabel('x')
orbit.set_ylabel('y')
orbit.set_title('First law: an ellipse with the star at a focus')
orbit.legend(loc='upper left')
area.plot(simulation.trajectory_.times, areal)
area.set_ylim(0, 1.2 * areal.max())
area.set_xlabel('time')
area.set_ylabel('areal velocity')
area.set_title('Second law: equal areas in equal times')
fig.tight_layout()
plt.show()
