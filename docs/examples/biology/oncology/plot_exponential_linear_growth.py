"""
# Exponential-linear growth

The tumour volume first grows exponentially, then bends into a straight line once its growth reaches the limiting rate
of 67.8 mm³ a day, near day 15.
"""

# %%
# The example builds `ExponentialLinearGrowthSystem` and runs it for 40 days from 2 mm³ with the breast cancer fit of
# Padmanabhan et al. The observer saves every frame with the default blueprint, whose primary view is the time series of
# the volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import ExponentialLinearGrowthSystem

SYSTEM = 'exponential_linear_growth'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ExponentialLinearGrowthSystem(n=2.0), observers=[observer])

# %%
# Growth rate
# -----------
# The growth rate `dn/dt` should rise in proportion to the volume, as `r0 n`, until it reaches `r` at
# `tau = log(r / (r0 n0)) / r0`, and stay constant after.

times, volume = trajectory.times, trajectory.get_values('tumour.x')
r0, r = 0.31, 67.8
tau = np.log(r / (r0 * 2.0)) / r0
fig, (curve, rate) = plt.subplots(1, 2, figsize=(10, 4))
curve.semilogy(times, volume)
curve.axvline(tau, color='grey', linestyle=':', label='switch at tau')
curve.set_xlabel('time (days)')
curve.set_ylabel('volume (mm³)')
curve.set_title('Straight on a log scale until tau')
curve.legend()
rate.plot(times, np.gradient(volume, times), label='simulated')
rate.plot(times, np.minimum(r0 * volume, r), '--', color='black', label='min(r0 n, r)')
rate.set_xlabel('time (days)')
rate.set_ylabel('growth rate (mm³ / day)')
rate.set_title('The growth rate saturates at r')
rate.legend()
fig.tight_layout()
plt.show()
