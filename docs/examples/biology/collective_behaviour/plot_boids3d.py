"""
# Boids in three dimensions

Boids fly in a periodic cube and steer by the boids they can see. With alignment alone the flock turns to one heading
while it keeps its spread in depth, so the third dimension stays in use.
"""

# %%
# Cohesion and separation are switched off to isolate alignment. The default blueprint shows the boids in a
# three-dimensional view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.collective_behaviour import Boids3dSystem

SYSTEM = 'boids3d'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(
    Boids3dSystem(n_boids=100, cohesion=0.0, separation=0.0, random_state=0),
    observers=[observer],
    random_state=0,
)

# %%
# Polarization and depth
# ----------------------
# The polarization, the length of the mean heading, rises from near zero to one as alignment spreads. The spread of
# the boids along the third axis stays near the side of the cube, as the boids do not collapse onto a plane.

polarization, depth = [], []
for frame in range(len(trajectory.times)):
    boids = trajectory.get_agents('flock', frame)
    headings = boids.values['velocity'] / np.linalg.norm(boids.values['velocity'], axis=1, keepdims=True)
    polarization.append(np.linalg.norm(headings.mean(axis=0)))
    depth.append(np.ptp(boids.locations[:, 2]))
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].plot(trajectory.times, polarization)
axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel('step')
axes[0].set_ylabel('polarization')
axes[0].set_title('Headings align')
axes[1].plot(trajectory.times, depth)
axes[1].set_ylim(0, 20)
axes[1].set_xlabel('step')
axes[1].set_ylabel('spread along the third axis')
axes[1].set_title('The flock keeps its depth')
plt.tight_layout()
plt.show()
