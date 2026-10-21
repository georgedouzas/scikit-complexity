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

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stormer_problem'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=20)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=100)

# %%
# Conserved quantities
# --------------------
# The magnetic force is perpendicular to the velocity, so the speed should stay at its initial value, and the symmetry
# about the dipole axis should conserve `x y' - y x' + alpha (x² + y²) / r³`. Both curves should stay flat to the
# tolerance of the integrator.

trajectory = simulation.trajectory_
x, y, z = (np.ravel(trajectory[f'charge.{name}']) for name in 'xyz')
vx, vy, vz = (np.ravel(trajectory[f'charge.{name}_d1']) for name in 'xyz')
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
