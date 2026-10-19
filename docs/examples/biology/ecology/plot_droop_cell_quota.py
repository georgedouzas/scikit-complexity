"""
# Droop cell-quota growth

The biomass first falls while the cells fill their nutrient quota, and grows once the quota is high enough for the
growth rate to exceed the death rate.
"""

# %%
# The model is created by name from the catalog: algae start at their minimum quota, where they cannot grow, in a
# medium whose uptake fills the quota to twice the minimum, with a death rate of 0.45 a day. The observer saves every
# frame with the default blueprint, whose primary view is the time series of the quota and the biomass.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'droop_cell_quota'

system = create(MODEL, Q=1.0, d=0.45)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# Quota equilibrium
# -----------------
# The quota settles at `Q* = q + v / r_inf = 2`, where the uptake `v = vmax R / (g + R) = 1` balances the dilution by
# growth. The biomass then grows exponentially at `r_inf (1 - q / Q*) - d = 0.05` a day.

trajectory = simulation.trajectory_
times = trajectory.times
fig, (quota, biomass) = plt.subplots(1, 2, figsize=(10, 4))
quota.plot(times, trajectory['algae.Q'], label='simulated')
quota.axhline(2.0, color='black', linestyle='--', label='Q*')
quota.set_xlabel('time (days)')
quota.set_ylabel('cell quota')
quota.set_title('The quota fills to its equilibrium')
quota.legend()
biomass.semilogy(times, trajectory['algae.P'], label='simulated')
late = times >= 10
exponential = trajectory['algae.P'][late][0] * np.exp(0.05 * (times[late] - times[late][0]))
biomass.semilogy(times[late], exponential, '--', color='black', label='rate 0.05')
biomass.set_xlabel('time (days)')
biomass.set_ylabel('biomass')
biomass.set_title('The biomass recovers and grows')
biomass.legend()
fig.tight_layout()
plt.show()
