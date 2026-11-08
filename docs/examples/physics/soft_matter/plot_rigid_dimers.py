"""
# Rigid Brownian and active dimers

Two spheres joined by a rigid rod form a colloidal molecule. Its centre diffuses and its rod turns randomly, and when
the spheres swim, the molecule dances: it drifts along the mean of their directions and turns with their difference.
"""

# %%
# A hundred passive dimers with rods of unit length start at the origin. The default blueprint shows the centres and
# the rod angles.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.soft_matter import RigidDimersSystem

SYSTEM = 'rigid_dimers'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = simulate(RigidDimersSystem(), until=20, observers=[observer], random_state=0)

# %%
# Rotational diffusion and active swimming
# ----------------------------------------
# The angular mean squared displacement of the rod should grow as `4 D_t t / L²`, so the rotational diffusion falls as
# `L^-2`, and active spheres should spread the centres much further than passive ones.

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
for length in (0.5, 1.0, 2.0):
    trajectory = simulate(RigidDimersSystem(L=length), until=2, random_state=0)
    times = np.asarray(trajectory.times)
    angles = np.asarray(trajectory.values('dimers.phi'))
    left.plot(times, ((angles - angles[0]) ** 2).mean(axis=1), label=f'L = {length}')
    left.plot(times, 4 * 0.1 * times / length**2, '--', color='grey')
left.set_xlabel('time')
left.set_ylabel('angular mean squared displacement')
left.legend()
for v0 in (0.0, 1.0):
    centres = np.asarray(simulate(RigidDimersSystem(v0=v0), until=20, random_state=0).values('dimers.c'))
    right.plot(centres[-1, :, 0], centres[-1, :, 1], '.', label=f'v0 = {v0}')
right.set_aspect('equal')
right.set_title('Centres after 20 time units')
right.legend()
plt.show()
