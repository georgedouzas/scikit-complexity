"""
# Relativistic orbit equation

The first relativistic correction to Newton's gravity adds `epsilon u²` to the equation of the inverse radius of a
planet. The ellipse no longer closes: its perihelion advances a little every revolution, and the orbit becomes a
rosette.
"""

# %%
# The correction `epsilon = 0.05` is exaggerated so that the advance is visible after a few turns. The simulation time
# is the polar angle of the planet, and the default blueprint shows the inverse radius `u` along it.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'relativistic_orbit'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
simulation = Simulation(system, dt=0.02, observers=[observer]).prepare().run(until=60)

# %%
# Precessing orbit
# ----------------
# The orbit in the plane is `(cos θ / u, sin θ / u)`. Without the correction it should be a closed ellipse, and with
# it a rosette whose perihelion turns by about `2π epsilon alpha` per revolution.

fig, ax = plt.subplots(figsize=(6, 6))
for epsilon in [0.0, 0.05]:
    run = Simulation(create_system(SYSTEM, epsilon=epsilon), dt=0.02).prepare().run(until=60)
    angle = np.asarray(run.trajectory_.times)
    radius = 1 / np.ravel(run.trajectory_.values('planet.u'))
    ax.plot(radius * np.cos(angle), radius * np.sin(angle), label=f'epsilon = {epsilon}')
ax.plot(0, 0, '*', color='orange', markersize=12)
ax.set_aspect('equal')
ax.set_title('Perihelion precession')
ax.legend()
plt.show()
