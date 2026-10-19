"""
# Power-law growth

The tumour volume grows more slowly than an exponential, since only a part of the tumour that scales as a power of its
volume proliferates.
"""

# %%
# The model is created by name from the catalog and run for 40 days from 10 mm³ with the breast cancer fit of
# Padmanabhan et al., `r = 1.32` and `p = 0.58`. The observer saves every frame with the default blueprint, whose
# primary view is the time series of the volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'power_law_growth'

system = create(MODEL, n=10.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=40)

# %%
# Growth rate against volume
# --------------------------
# The model `dn/dt = r n^p` gives a straight line of slope `p` between `log(dn/dt)` and `log(n)`, and the volume
# follows `(n0^(1-p) + (1-p) r t)^(1/(1-p))`.

times, volume = simulation.trajectory_.times, simulation.trajectory_['tumour.n']
r, p = 1.32, 0.58
exact = (10.0 ** (1 - p) + (1 - p) * r * times) ** (1 / (1 - p))
fig, (curve, scaling) = plt.subplots(1, 2, figsize=(10, 4))
curve.plot(times, volume, label='simulated')
curve.plot(times, exact, '--', color='black', label='closed form')
curve.set_xlabel('time (days)')
curve.set_ylabel('volume (mm³)')
curve.set_title('The volume follows the closed form')
curve.legend()
scaling.loglog(volume, np.gradient(volume, times), label='simulated')
scaling.loglog(volume, r * volume**p, '--', color='black', label='r n^p')
scaling.set_xlabel('volume (mm³)')
scaling.set_ylabel('growth rate (mm³ / day)')
scaling.set_title('The growth rate scales as n^0.58')
scaling.legend()
fig.tight_layout()
plt.show()
