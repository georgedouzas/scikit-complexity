"""
# Active polymers

A chain of beads bonded by springs, where every bead swims along its own orientation. The chain wriggles, swells
beyond the size of a passive chain, and its centre of mass wanders like a slow active particle.
"""

# %%
# A chain of twenty beads swims at `v0 = 1` with the bond stiffness `k = 10`. The default blueprint shows the bead
# coordinates.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import ActivePolymerSystem

SYSTEM = 'active_polymer'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = simulate(ActivePolymerSystem(), until=50, observers=[observer], random_state=0)

# %%
# Swelling with activity
# ----------------------
# Snapshots of a passive and an active chain after 50 time units, and the mean squared end-to-end distance against
# the swimming speed, which should start at the Rouse value `2 D_t (N - 1) / k`, dashed, and grow with the speed.

speeds = [0.0, 0.5, 1.0, 1.5, 2.0]
sizes = []
fig, (shapes, swelling) = plt.subplots(1, 2, figsize=(10, 4))
for v0 in speeds:
    trajectory = simulate(ActivePolymerSystem(v0=v0), until=200, random_state=1)
    x, y = np.asarray(trajectory.get_values('chain.x')), np.asarray(trajectory.get_values('chain.y'))
    sizes.append(np.mean(((x[:, -1] - x[:, 0]) ** 2 + (y[:, -1] - y[:, 0]) ** 2)[40:]))
    if v0 in (0.0, 2.0):
        shapes.plot(x[-1] - x[-1].mean(), y[-1] - y[-1].mean(), 'o-', markersize=3, label=f'v0 = {v0}')
shapes.set_aspect('equal')
shapes.legend()
shapes.set_title('Chains after 200 time units')
swelling.plot(speeds, sizes, 'o-')
swelling.axhline(2 * 0.1 * 19 / 10, color='black', linestyle='--')
swelling.set_xlabel('swimming speed v0')
swelling.set_ylabel('mean squared end-to-end distance')
plt.show()
