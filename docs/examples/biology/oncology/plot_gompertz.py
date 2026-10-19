"""
# Gompertz growth

The tumour volume rises along an S-shaped curve: fast at first, then ever slower as it approaches its carrying
capacity of 1964 mm³.
"""

# %%
# The model is created by name from the catalog and run for 100 days from a volume of 10 mm³, the breast cancer fit of
# Padmanabhan et al. The observer saves every frame with the default blueprint, whose primary view is the time series
# of the volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'gompertz'

system = create(MODEL, n=10.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# Closed-form solution
# --------------------
# The Gompertz law `dn/dt = r n log(K / n)` has the solution `n(t) = K exp(log(n0 / K) exp(-r t))`. The simulated
# volume should lie on it, and `log(log(K / n))` should fall on a straight line of slope `-r`.

times, volume = simulation.trajectory_.times, simulation.trajectory_['tumour.n']
r, K = 0.0719, 1964.0
exact = K * np.exp(np.log(10.0 / K) * np.exp(-r * times))
fig, (curve, line) = plt.subplots(1, 2, figsize=(10, 4))
curve.plot(times, volume, label='simulated')
curve.plot(times, exact, '--', color='black', label='closed form')
curve.axhline(K, color='grey', linestyle=':', label='carrying capacity')
curve.set_xlabel('time (days)')
curve.set_ylabel('volume (mm³)')
curve.set_title('The volume follows the Gompertz curve')
curve.legend()
line.plot(times, np.log(np.log(K / volume)), label='simulated')
line.plot(times, np.log(np.log(K / 10.0)) - r * times, '--', color='black', label='slope -r')
line.set_xlabel('time (days)')
line.set_ylabel('log log(K / n)')
line.set_title('The double logarithm falls linearly')
line.legend()
fig.tight_layout()
plt.show()
