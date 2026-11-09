"""
# Geodynamo reversals

The dipolar field `D` stays in one polarity for a while, oscillating around it, then flips to the other polarity at
an irregular time, as the magnetic field of the Earth has reversed throughout its history.
"""

# %%
# The system is built from its class with Gissinger's parameters `mu = 0.119`, `nu = 0.1`, and
# `Gamma = 0.9`. The run covers 600 time units, and the default blueprint shows the quadrupolar field `Q`, the dipolar
# field `D`, and the velocity `V` in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import GeodynamoSystem

SYSTEM = 'geodynamo'

system = GeodynamoSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=5)
trajectory = simulate(system, until=600, observers=[observer])

# %%
# Two polarities and irregular reversals
# --------------------------------------
# The equations are symmetric under `(Q, D) -> (-Q, -D)`, so the chaotic attractor should have two lobes of opposite
# polarity. In a long run, counting a reversal when `|D|` exceeds 0.5 with the opposite sign, the times spent in one
# polarity should be irregular, with a broad distribution of durations.

run = simulate(GeodynamoSystem(), until=20000)
settled = run.times > 500
times, Q, D = run.times[settled], run.get_values('dynamo.Q')[settled], run.get_values('dynamo.D')[settled]
strong = np.abs(D) > 0.5
polarity = np.sign(D[strong])
flips = times[strong][1:][polarity[1:] != polarity[:-1]]
durations = np.diff(flips)
fig, (attractor, intervals) = plt.subplots(1, 2, figsize=(10, 4))
attractor.plot(Q[:4000], D[:4000], linewidth=0.3)
attractor.set_xlabel('quadrupole Q')
attractor.set_ylabel('dipole D')
attractor.set_title('Two lobes of opposite polarity')
intervals.hist(durations, bins=np.logspace(np.log10(durations.min()), np.log10(durations.max()), 25))
intervals.set_xscale('log')
intervals.set_xlabel('time between reversals')
intervals.set_ylabel('count')
intervals.set_title(f'{len(durations)} polarity intervals')
fig.tight_layout()
plt.show()
