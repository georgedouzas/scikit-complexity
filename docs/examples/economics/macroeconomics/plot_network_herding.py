"""
# Linear network herding

Every agent's choice, such as its consumption, is a weighted sum of the choices of the agents it listens to in the
previous period plus a response to its own characteristic. Aggregating the choices gives a mean that depends on the
out-degrees of the network, so direct interaction makes the aggregate persistent.
"""

# %%
# Fifty agents on a random directed network whose weights add up to 0.5 for every agent receive shocks of standard
# deviation 0.1 on lognormal characteristics, for 50 periods. The default blueprint shows the mean choice.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.macroeconomics import NetworkHerdingSystem

SYSTEM = 'network_herding'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(NetworkHerdingSystem(), observers=[observer])

# %%
# Persistence and herding
# -----------------------
# With shocks of unit size, the autocorrelation of the mean choice should grow with the strength of the links, since
# Eq. 1.12 carries a share of last period's choices into the present through the out-degrees.

strengths = [0.0, 0.2, 0.4, 0.6, 0.8]
correlations = []
for herding in strengths:
    mean = simulate(NetworkHerdingSystem(sigma=1.0, herding=herding, density=0.3), until=2000).get_values(
        'mean_choice',
    )[100:]
    correlations.append(np.corrcoef(mean[:-1], mean[1:])[0, 1])
fig, (paths, persistence) = plt.subplots(1, 2, figsize=(10, 4))
paths.plot(trajectory.times, trajectory.get_values('agents.c')[:, :10], lw=0.8)
paths.plot(trajectory.times, trajectory.get_values('mean_choice'), 'k', lw=2, label='mean')
paths.set_xlabel('period')
paths.set_ylabel('choice c')
paths.set_title('Choices of ten agents and the mean')
paths.legend()
persistence.plot(strengths, correlations, 'o-')
persistence.set_xlabel('sum of the weights of an agent')
persistence.set_ylabel('lag-1 autocorrelation of the mean choice')
persistence.set_title('Herding makes the aggregate persistent')
fig.tight_layout()
plt.show()
