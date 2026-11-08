"""
# Nuclear quadrupole vibrations

The surface of a nucleus vibrates in a quartic potential with the threefold symmetry of the Hénon-Heiles potential.
At intermediate energies almost every orbit is chaotic, and its Poincaré section fills the allowed region.
"""

# %%
# The nucleus starts at `q = (0, 1)` with the momentum `p = (0.5, 0)`, at the energy `E = 0.75`, and the strength
# of the cubic term is `B = 0.5`. The default blueprint draws the path of the deformation in the `(q1, q2)` plane.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import section
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.nonlinear_oscillators import NuclearQuadrupoleSystem

SYSTEM = 'nuclear_quadrupole'
ENERGY = 0.75


def start(q2, p2, energy=ENERGY):
    """Return the quadrupole system started on `q1 = 0` at a given energy."""
    p1 = np.sqrt(2 * energy - p2**2 - q2**2 - q2**4 / 4)
    return NuclearQuadrupoleSystem(q1=0.0, q2=q2, p1=p1, p2=p2)


observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=8)
trajectory = simulate(start(1.0, 0.0), until=100, observers=[observer])

# %%
# Poincaré section
# ----------------
# Each orbit is recorded at its upward crossings of `q1 = 0`. Regular orbits on invariant tori would trace closed
# curves, while chaotic orbits scatter their points, and at this energy the scattered points should fill the whole
# region the energy allows.

fig, ax = plt.subplots(figsize=(6, 5))
for q2 in np.linspace(-1.0, 1.0, 9):
    points = section(start(q2, 0.0), state='nucleus.q1', value=0.0, until=800)
    ax.plot(points.values['nucleus.q2'], points.values['nucleus.p2'], '.', markersize=1.5)
ax.set_xlabel('q2')
ax.set_ylabel('p2')
ax.set_title('Poincaré section at q1 = 0, E = 0.75')
plt.show()
