"""
# Shepherding

A shepherd brings a scattered flock to a goal by alternating two moves: it collects the sheep furthest from the flock,
and when the flock is compact, drives it from behind towards the goal. Sheep flee the shepherd while they keep together
and apart, so the flock moves as one.
"""

# %%
# The system is built from its class with thirty sheep, and the observer saves every step with the default blueprint,
# whose primary view is the field with the sheep and the shepherd.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.collective_behaviour import ShepherdingSystem

SYSTEM = 'shepherding'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ShepherdingSystem(), observers=[observer], random_state=0, until=200)

# %%
# Path of the flock to the goal
# -----------------------------
# The centre of the flock should approach the goal at `(5, 5)`, while the shepherd zig-zags behind it.

centres = np.array([np.asarray(trajectory.agents('sheep', f).locations, float).mean(axis=0) for f in range(201)])
dog = np.array([np.asarray(trajectory.agents('shepherds', f).locations, float)[0] for f in range(201)])
fig, (field, distance) = plt.subplots(1, 2, figsize=(11, 4.5))
field.plot(*centres.T, label='centre of the flock')
field.plot(*dog.T, lw=0.6, label='shepherd')
field.plot(5, 5, '*', markersize=14, color='black', label='goal')
field.set_xlim(0, 100)
field.set_ylim(0, 100)
field.set_aspect('equal')
field.set_title('Paths in the field')
field.legend(fontsize=8)
distance.plot(np.linalg.norm(centres - 5.0, axis=1))
distance.set_xlabel('step')
distance.set_ylabel('distance of the flock to the goal')
distance.set_title('The flock reaches the goal')
fig.tight_layout()
plt.show()
