"""
# Hénon-Heiles

A star orbiting in a triangular galactic potential: at low energy its orbit is a regular rosette, while at energy
`1 / 8` and above many orbits wander chaotically, and the Poincaré section shows islands of order in a sea of chaos.
"""

# %%
# The star starts on the line `x = 0` with the energy `E = 1 / 8`. The default blueprint draws the whole path
# of the star in the plane.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import poincare_section
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'henon_heiles'
ENERGY = 1 / 8


def start(y, py, energy=ENERGY):
    """Return the Hénon-Heiles model started on `x = 0` at a given energy."""
    px = np.sqrt(2 * energy - py**2 - y**2 + 2 * y**3 / 3)
    return create(MODEL, x=0.0, y=y, px=px, py=py)


observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = Simulation(start(-0.1, 0.0), dt=0.05, observers=[observer]).run(until=100)

# %%
# Poincaré section
# ----------------
# Each orbit is recorded at its upward crossings of `x = 0`. Regular orbits lie on invariant tori and should trace
# closed curves around islands, while chaotic orbits should scatter over a connected sea between them.

fig, ax = plt.subplots(figsize=(6, 5))
for y in np.linspace(-0.3, 0.45, 12):
    section = poincare_section(start(y, 0.0), 'star.x', 0.0, until=2000)
    ax.plot(section['star.y'], section['star.y_d1'], '.', markersize=1.5)
ax.set_xlabel('y')
ax.set_ylabel('p_y')
ax.set_title('Poincaré section at x = 0, E = 1/8')
plt.show()
