"""
# CCC model

Entities, such as goods or technologies, exist when the pairs of existing entities that produce them outnumber the
pairs that destroy them. Updated one by one with rare spontaneous changes, the system rests for long periods in one
composition and then reorganises in bursts, the punctuated equilibria of combinatorial evolution.
"""

# %%
# A hundred entities with on average five productive and ten destructive pairs each, and spontaneous flips with
# probability 0.0002, run for 1000 steps. The default blueprint shows the diversity and the changes per step.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.innovation import CccModelSystem

SYSTEM = 'ccc_model'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(CccModelSystem(), observers=[observer])

# %%
# Punctuated equilibria
# ---------------------
# The states of the entities over time should show horizontal stripes of quasi-equilibria broken by bursts of change,
# as in Fig. 5.14 of Thurner et al., and the changes per step should be zero most of the time with rare large values.

states = np.array([trajectory.agents('entities', frame).values['exists'] for frame in range(len(trajectory.times))])
changes = trajectory.values('update.changes')[1:]
fig, (raster, sizes) = plt.subplots(1, 2, figsize=(10, 4))
raster.imshow(states.T, aspect='auto', cmap='gray', interpolation='nearest')
raster.set_xlabel('step')
raster.set_ylabel('entity')
raster.set_title('Existing entities (white)')
sizes.hist(changes[changes > 0], bins=np.arange(1, changes.max() + 2) - 0.5, log=True)
sizes.set_xlabel('entities changed in a step')
sizes.set_ylabel('steps')
sizes.set_title(f'{np.mean(changes == 0):.0%} of the steps change nothing')
fig.tight_layout()
plt.show()
