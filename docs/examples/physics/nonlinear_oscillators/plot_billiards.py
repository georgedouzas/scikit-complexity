"""
# Dynamical billiards of Sinai, mushroom, and open three-disk shapes

A particle bounces around a square table with a round obstacle in the middle. The convex disc scatters nearby paths
apart at every hit, so the motion of the Sinai billiard is chaotic and its path soon criss-crosses the whole table. In
the open three-disk billiard the particle comes from far away, bounces between three discs, and leaves again.
"""

# %%
# The default blueprint draws the table and the whole path of the particle.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.nonlinear_oscillators import BilliardsSystem

SYSTEM = 'billiards'


system = BilliardsSystem(shape='sinai', x=0.1, y=0.6, vx=0.6, vy=0.8)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(system, until=40, dt=0.04, observers=[observer])

# %%
# Separation of nearby paths
# --------------------------
# A second particle starts with its velocity turned by `10^-8` radians. In a chaotic billiard the distance between
# the two particles should grow exponentially, a straight line on the logarithmic axis, until it saturates at the size
# of the table.

turn = 1e-8
vx, vy = 0.6 * np.cos(turn) - 0.8 * np.sin(turn), 0.6 * np.sin(turn) + 0.8 * np.cos(turn)
paths = [
    (simulate(BilliardsSystem(shape='sinai', x=0.1, y=0.6, vx=velocity[0], vy=velocity[1]), until=60, dt=0.01))
    for velocity in [(0.6, 0.8), (vx, vy)]
]
times = np.asarray(paths[0].times)
distance = np.hypot(
    np.ravel(paths[0].get_values('particle.x')) - np.ravel(paths[1].get_values('particle.x')),
    np.ravel(paths[0].get_values('particle.y')) - np.ravel(paths[1].get_values('particle.y')),
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(times, distance)
ax.set_xlabel('time')
ax.set_ylabel('distance between the particles')
ax.set_title('Exponential divergence in the Sinai billiard')
plt.show()

# %%
# Chaotic scattering by three discs
# ---------------------------------
# Particles come from far away towards the center of the three-disk table, from the polar angle `phi`, and the
# number of their collisions before they leave is counted. Most directions give one or two collisions, but near the
# edges of the discs the count should jump, and every zoom into such an edge should show more of these jumps, the
# fractal scattering function.

angles = np.linspace(1.44, 1.56, 121)
collisions = []
for phi in angles:
    system = BilliardsSystem(
        shape='three_disk',
        x=8 * np.cos(phi),
        y=8 * np.sin(phi),
        vx=-np.cos(phi),
        vy=-np.sin(phi),
    )
    collisions.append(len(simulate(system, until=25, dt=0.1).events))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(angles, collisions, '.-')
ax.set_xlabel('incoming angle phi')
ax.set_ylabel('number of collisions')
ax.set_title('Scattering by the open three-disk billiard')
plt.show()
