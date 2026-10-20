"""
# Continuous-time random walk

The walkers jump by Gaussian steps after waiting times with no mean. Most walkers jump often, but some fall into long
waits and stay put, so the cloud spreads slower and slower and keeps a sharp peak at the origin.
"""

# %%
# The model is created by name from the catalog with 500 walkers in the plane, whose waiting times have the tail
# `tau^(-3/2)`, and simulated to time 1000. Every walker renews on its own clock and jumps at its renewals, with no
# time grid. The observer saves every frame with the default blueprint, whose primary view shows the walkers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'continuous_time_random_walk'

system = create(MODEL, n_walkers=500, alpha=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=5, observers=[observer], random_state=0).run(until=1000)

# %%
# Subdiffusion from waiting times without a mean
# ----------------------------------------------
# With waiting times of tail exponent `alpha = 0.5`, the number of jumps up to time `t` grows as `t^alpha`, and so does
# the ensemble mean squared displacement, since every jump adds the same variance. Both curves should follow the
# dashed lines of slope 0.5 on log-log axes, well below the slope 1 of Brownian motion.

trajectory = simulation.trajectory_
times = trajectory.times[1:]
squared = np.mean(np.sum(trajectory['walkers.x'][1:] ** 2, axis=-1), axis=1)
jumps = trajectory['jumps.renewals'][1:] / 500
slope = np.polyfit(np.log(times), np.log(squared), 1)[0]
fig, (displacement, counts) = plt.subplots(1, 2, figsize=(10, 4))
displacement.loglog(times, squared, label=f'simulation, slope {slope:.2f}')
displacement.loglog(times, squared[-1] * (times / times[-1]) ** 0.5, '--', color='black', label='slope 0.5')
displacement.set_xlabel('time t')
displacement.set_ylabel('ensemble mean squared displacement')
displacement.set_title('Subdiffusion, <x^2> ~ t^alpha')
displacement.legend()
counts.loglog(times, jumps, label='mean number of jumps')
counts.loglog(times, jumps[-1] * (times / times[-1]) ** 0.5, '--', color='black', label='slope 0.5')
counts.set_xlabel('time t')
counts.set_ylabel('jumps per walker')
counts.set_title('The jumps grow as t^alpha')
counts.legend()
fig.tight_layout()
plt.show()
