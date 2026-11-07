"""
# Störmer charge in a dipole field

A charged particle near a magnetized planet spirals along the field lines, bounces between mirror points in the north
and the south, and drifts slowly around the planet. Such trapped orbits form the radiation belts.
"""

# %%
# The particle starts one planet radius from the centre in the equatorial plane, moving slowly along and across the
# field. The default blueprint draws its whole path in three dimensions around the planet, marked at the origin.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.celestial_mechanics import StormerProblemSystem

SYSTEM = 'stormer_problem'

system = StormerProblemSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=20)
trajectory = simulate(system, until=100, observers=[observer])

# %%
# Conserved quantities
# --------------------
# The magnetic force is perpendicular to the velocity, so the speed should stay at its initial value, and the symmetry
# about the dipole axis should conserve `x y' - y x' + alpha (x² + y²) / r³`. Both curves should stay flat to the
# tolerance of the integrator.

x, y, z = (np.ravel(trajectory.values(f'charge.{name}')) for name in 'xyz')
vx, vy, vz = (np.ravel(trajectory.values(f'charge.{name}_d1')) for name in 'xyz')
speed = np.sqrt(vx**2 + vy**2 + vz**2)
momentum = x * vy - y * vx + (x**2 + y**2) / (x**2 + y**2 + z**2) ** 1.5
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
top.plot(trajectory.times, speed / speed[0] - 1)
top.set_ylabel('relative change of speed')
top.set_title('The magnetic force does no work')
bottom.plot(trajectory.times, momentum - momentum[0])
bottom.set_xlabel('time')
bottom.set_ylabel('change of the axial invariant')
fig.tight_layout()
plt.show()
