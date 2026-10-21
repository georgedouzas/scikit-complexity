"""
# Nervous rats

Nervousness spreads among rats by contact and fades on its own. When contagion is stronger than calming, a few nervous
rats start an outbreak and the colony settles with a share `1 - calming / contagion` of rats that stay nervous.
"""

# %%
# The model runs as a jump process of single rats becoming nervous or calm, with a hundred rats, five of them nervous,
# and contagion twice the calming. The observer saves every frame with the default blueprint, whose primary view is
# the time series of the calm and the nervous rats.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'rats'

system = create(MODEL, n_rats=100, nervous=5, contagion=2.0, calming=1.0, spontaneous=0.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer], random_state=0).run(until=20)

# %%
# The phase transition
# --------------------
# Without spontaneous alarm, the long-run share of nervous rats should be zero below `contagion = calming` and
# `1 - calming / contagion` above it. The deterministic mode of the model should trace that curve.

contagions = np.linspace(0.25, 4.0, 16)
shares = []
for contagion in contagions:
    system = create(MODEL, contagion=contagion, calming=1.0, spontaneous=0.0, stochastic=False)
    trajectory = Simulation(system, dt=0.1).run(until=60).trajectory_
    shares.append(trajectory['nervous.n'][-1] / 100)
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
