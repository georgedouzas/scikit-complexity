"""
# Mangrove succession

After a clear cut of a Sundarbans mangrove forest, the pioneer Keora regrows first, the intermediate Gewa follows,
and the slow climax tree Sundari finally overtakes both and suppresses them, the classic sequence of an ecological
succession.
"""

# %%
# The system is built from its class from a bare clear cut and run for 300 years. The observer saves every year with
# the default blueprint, whose primary view is the time series of the three volumes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import MangroveSuccessionSystem

SYSTEM = 'mangrove_succession'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(MangroveSuccessionSystem(), observers=[observer], until=299)

# %%
# The order of dominance
# ----------------------
# The share of the total volume held by each group shows the succession: Keora should dominate first, Gewa should
# peak later, and Sundari should hold most of the mature forest.

groups = ['keora', 'gewa', 'sundari']
volumes = np.column_stack([trajectory.values(f'{group}.volume') for group in groups])
shares = volumes / np.maximum(volumes.sum(axis=1, keepdims=True), 1e-12)
peaks = trajectory.times[np.argmax(volumes, axis=0)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.stackplot(
    trajectory.times,
    shares.T,
    labels=[f'{group}, peak at year {peak:.0f}' for group, peak in zip(groups, peaks, strict=True)],
)
ax.set_xlabel('years after the clear cut')
ax.set_ylabel('share of the standing volume')
ax.set_title(f'Sundari holds {shares[-1, 2]:.0%} of the mature forest')
ax.legend(loc='center right')
plt.show()
