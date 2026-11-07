"""
# Generalized logical network

Multi-valued variables step one level at a time towards the focal levels their regulators set, as random asynchronous
events. A negative circuit runs around a cycle of its state graph, a sustained oscillation, while a positive circuit
stops in one of two stable states.
"""

# %%
# The system is created by name from the catalog with the negative circuit, and the observer saves every frame with the
# default blueprint, whose primary view is the time series of the levels.

import collections
import itertools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'generalized_logical_network'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.25, observers=[observer], random_state=0).prepare().run(until=50)

# %%
# The asynchronous state graph
# ----------------------------
# The transitions between consecutive frames should trace a cycle through the states of `x` and `y`.

trajectory = simulation.trajectory_
states = list(zip(trajectory.values('x.level').astype(int), trajectory.values('y.level').astype(int), strict=True))
moves = collections.Counter((a, b) for a, b in itertools.pairwise(states) if a != b)
fig, ax = plt.subplots(figsize=(6, 4.5))
for (start, end), count in moves.items():
    ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '->', 'lw': 0.5 + count / 3})
visits = collections.Counter(states)
ax.scatter(*np.array(list(visits)).T, s=[20 * count for count in visits.values()], zorder=3)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1])
ax.set_xlabel('level of x')
ax.set_ylabel('level of y')
ax.set_title('Visited states and transitions of the negative circuit')
plt.show()
