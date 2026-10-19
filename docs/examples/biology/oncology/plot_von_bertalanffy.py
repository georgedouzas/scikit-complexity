"""
# Von Bertalanffy growth

The tumour volume rises and levels off where its growth, which scales as a power of the volume, balances its loss,
which scales with the volume.
"""

# %%
# The model is created by name from the catalog with the exponent `p = 2/3` of a proliferating rim and run for 40 days
# from 1 mm³. The observer saves every frame with the default blueprint, whose primary view is the time series of the
# volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'von_bertalanffy'

r, p, d = 2.32, 2 / 3, 0.808
system = create(MODEL, n=1.0, r=r, p=p, d=d)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=40)

# %%
# Closed-form solution
# --------------------
# The model `dn/dt = r n^p - d n` has the solution `(r / d + (n0^(1-p) - r / d) exp(-d (1-p) t))^(1/(1-p))`, which
# tends to the equilibrium `(r / d)^(1/(1-p))`. The simulated volume should lie on it.

times, volume = simulation.trajectory_.times, simulation.trajectory_['tumour.n']
exact = (r / d + (1.0 - r / d) * np.exp(-d * (1 - p) * times)) ** (1 / (1 - p))
fig, (curve, error) = plt.subplots(1, 2, figsize=(10, 4))
curve.plot(times, volume, label='simulated')
curve.plot(times, exact, '--', color='black', label='closed form')
curve.axhline((r / d) ** (1 / (1 - p)), color='grey', linestyle=':', label='equilibrium')
curve.set_xlabel('time (days)')
curve.set_ylabel('volume (mm³)')
curve.set_title('The volume levels off at the equilibrium')
curve.legend()
error.semilogy(times[1:], np.abs(volume[1:] / exact[1:] - 1))
error.set_xlabel('time (days)')
error.set_ylabel('relative error')
error.set_title('The simulation agrees with the closed form')
fig.tight_layout()
plt.show()
