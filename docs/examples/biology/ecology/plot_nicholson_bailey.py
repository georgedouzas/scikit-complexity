"""
# Nicholson-Bailey

Hosts and the parasitoids that lay their eggs in them oscillate from generation to generation with an ever larger
amplitude, because their equilibrium is unstable: the populations spiral away from it until they crash.
"""

# %%
# The model is created by name from the catalog and started close to its equilibrium of about 27.7 hosts and 13.9
# parasitoids. The observer saves every generation with the default blueprint, whose primary view is the time series
# of both populations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'nicholson_bailey'

system = create(MODEL, hosts=25.0, parasitoids=12.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=30)

# %%
# An unstable spiral
# ------------------
# The equilibrium is `H = k ln k / ((k - 1) a c)` and `P = ln k / a`. In the phase plane the generations should
# spiral outwards from it, until a crash leaves almost no hosts and the parasitoids starve.

hosts, parasitoids = simulation.trajectory_['hosts.n'], simulation.trajectory_['parasitoids.n']
before_crash = np.cumprod((hosts > 1) & (parasitoids > 1)).astype(bool)
hosts, parasitoids = hosts[before_crash], parasitoids[before_crash]
k, a, c = 2.0, 0.05, 1.0
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(hosts, parasitoids, 'o-', markersize=3, label='generations')
ax.plot(k * np.log(k) / ((k - 1) * a * c), np.log(k) / a, '*', color='black', markersize=12, label='equilibrium')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('hosts')
ax.set_ylabel('parasitoids')
ax.set_title(f'Generations spiral away from the equilibrium, crash after {before_crash.sum()} generations')
ax.legend()
plt.show()
