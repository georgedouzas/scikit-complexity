"""
# Sitnikov problem

A light body bobs up and down through the centre of two equal stars that orbit each other. With circular orbits its
motion is a regular oscillation, but when the stars move on ellipses the bobbing becomes irregular: regular islands
coexist with a chaotic sea.
"""

# %%
# The stars move on ellipses of eccentricity `e = 0.2`, and the body starts at rest at height 0.9, in the chaotic band.
# The default blueprint shows its height and velocity, whose oscillations change their amplitude irregularly.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import stroboscopic
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sitnikov'

system = create(MODEL, z=0.9, vz=0.0, e=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=50)

# %%
# Stroboscopic section
# --------------------
# Sampling the body once per period `2π` of the stars gives a map of the plane `(z, z')`. For `e = 0.2` orbits started
# close to the centre should trace closed invariant curves around the fixed point at the origin, and orbits started
# higher up should scatter over a chaotic band, some of them escaping.

fig, ax = plt.subplots(figsize=(6, 5))
for height in np.linspace(0.1, 1.5, 12):
    section = stroboscopic(create(MODEL, z=height, vz=0.0, e=0.2), 2 * np.pi, n_points=300, transient=0)
    z, velocity = section['body.z'], section['body.z_d1']
    near = np.abs(z) < 3
    ax.plot(z[near], velocity[near], '.', markersize=1.5)
ax.set_xlabel('z')
ax.set_ylabel("z'")
ax.set_title('Stroboscopic section of the Sitnikov problem, e = 0.2')
plt.show()
