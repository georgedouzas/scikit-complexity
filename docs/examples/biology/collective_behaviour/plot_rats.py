"""
# Nervous rats

Nervousness spreads among rats by contact and fades on its own. When contagion is stronger than calming, a few nervous
rats start an outbreak and the colony settles with a share `1 - calming / contagion` of rats that stay nervous.
"""

# %%
# The system runs as a jump process of single rats becoming nervous or calm, with a hundred rats, five of them nervous,
# and contagion twice the calming. The observer saves every frame with the default blueprint, whose primary view is
# the time series of the calm and the nervous rats.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.collective_behaviour import RatsSystem

SYSTEM = 'rats'

system = RatsSystem(n_rats=100, nervous=5, contagion=2.0, calming=1.0, spontaneous=0.0).prepare().system_
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, dt=0.1, observers=[observer], random_state=0, until=20)

# %%
# The phase transition
# --------------------
# Without spontaneous alarm, the long-run share of nervous rats should be zero below `contagion = calming` and
# `1 - calming / contagion` above it. The deterministic mode of the system should trace that curve.

contagions = np.linspace(0.25, 4.0, 16)
shares = []
for contagion in contagions:
    system = RatsSystem(contagion=contagion, calming=1.0, spontaneous=0.0, stochastic=False).prepare().system_
    trajectory = simulate(system, dt=0.1, until=60)
    shares.append(trajectory.values('nervous.n')[-1] / 100)
grid = np.linspace(0.25, 4.0, 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(contagions, shares, 'o', label='deterministic mode')
ax.plot(grid, np.clip(1 - 1 / grid, 0, None), '--', color='black', label='1 - calming / contagion')
ax.axvline(1.0, color='grey', linestyle=':', label='transition')
ax.set_xlabel('contagion / calming')
ax.set_ylabel('share of nervous rats')
ax.set_title('Nervousness persists above the transition')
ax.legend()
plt.show()
