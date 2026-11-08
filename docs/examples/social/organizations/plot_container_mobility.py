"""
# Container model of human mobility

Alessandretti, Aslak, and Lehmann describe space as a hierarchy of containers, such as buildings in neighbourhoods in
cities. An individual chooses first at which level to travel and then a destination by the attractiveness of the
containers, so its displacements mix the typical sizes of the containers, the scales of human mobility.
"""

# %%
# Ten individuals move in a nested grid of three levels, every container holding four by four containers of the level
# below, and travel at the three levels with the probabilities 0.6, 0.3, and 0.1. The default blueprint shows their
# positions and the level of their last displacement.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.organizations import ContainerMobilitySystem

SYSTEM = 'container_mobility'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(ContainerMobilitySystem(), observers=[observer], random_state=0)

# %%
# Displacements at three scales
# -----------------------------
# The lengths of the displacements should split into three bumps, one per level, near one, four, and sixteen cells,
# and the shares of the levels should match the probabilities of travel.

frames = range(1, len(trajectory.times))
x = np.array([trajectory.agents('individuals', frame).values['x'] for frame in frames])
y = np.array([trajectory.agents('individuals', frame).values['y'] for frame in frames])
levels = np.array([trajectory.agents('individuals', frame).values['level'] for frame in frames])[1:]
lengths = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
fig, (scales, shares) = plt.subplots(1, 2, figsize=(10, 4))
bins = np.logspace(-0.1, 2, 40)
for level in (1, 2, 3):
    scales.hist(lengths[levels == level], bins=bins, alpha=0.6, label=f'level {level}')
scales.set_xscale('log')
scales.set_xlabel('displacement length')
scales.set_ylabel('displacements')
scales.set_title('A mixture of scales')
scales.legend()
shares.bar([1, 2, 3], [np.mean(levels == level) for level in (1, 2, 3)], label='simulation')
shares.plot([1, 2, 3], [0.6, 0.3, 0.1], 'o', color='black', label='probabilities of travel')
shares.set_xlabel('level-distance')
shares.set_ylabel('share of displacements')
shares.set_title('Levels of travel')
shares.legend()
fig.tight_layout()
plt.show()
