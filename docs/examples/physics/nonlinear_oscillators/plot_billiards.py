"""
# Dynamical billiards of Sinai and mushroom shapes

A particle bounces around a square table with a round obstacle in the middle. The convex disc scatters nearby paths
apart at every hit, so the motion of the Sinai billiard is chaotic and its path soon criss-crosses the whole table.
"""

# %%
# The default blueprint draws the table and the whole path of the particle.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'billiards'


system = create(MODEL, shape='sinai', x=0.1, y=0.6, vx=0.6, vy=0.8)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.04, observers=[observer]).run(until=40)

# %%
# Separation of nearby paths
# --------------------------
# A second particle starts with its velocity turned by `10^-8` radians. In a chaotic billiard the distance between
# the two particles should grow exponentially, a straight line on the logarithmic axis, until it saturates at the size
# of the table.

turn = 1e-8
vx, vy = 0.6 * np.cos(turn) - 0.8 * np.sin(turn), 0.6 * np.sin(turn) + 0.8 * np.cos(turn)
paths = [
    Simulation(create(MODEL, shape='sinai', x=0.1, y=0.6, vx=velocity[0], vy=velocity[1]), dt=0.01).run(until=60)
    for velocity in [(0.6, 0.8), (vx, vy)]
]
times = np.asarray(paths[0].trajectory_.times)
distance = np.hypot(
    np.ravel(paths[0].trajectory_['particle.x']) - np.ravel(paths[1].trajectory_['particle.x']),
    np.ravel(paths[0].trajectory_['particle.y']) - np.ravel(paths[1].trajectory_['particle.y']),
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(times, distance)
ax.set_xlabel('time')
ax.set_ylabel('distance between the particles')
ax.set_title('Exponential divergence in the Sinai billiard')
plt.show()
