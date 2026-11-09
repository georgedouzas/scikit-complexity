"""
# Von Bertalanffy growth

The tumour volume rises and levels off where its growth, which scales as a power of the volume, balances its loss,
which scales with the volume.
"""

# %%
# The system is built from its class with the exponent `p = 2/3` of a proliferating rim and run for 40 days from 1
# mm³. The observer saves every frame with the default blueprint, whose primary view is the time series of the volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import VonBertalanffySystem

SYSTEM = 'von_bertalanffy'

r, p, d = 2.32, 2 / 3, 0.808
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(VonBertalanffySystem(n=1.0, r=r, p=p, d=d), dt=0.2, observers=[observer], until=40)

# %%
# Closed-form solution
# --------------------
# The system `dn/dt = r n^p - d n` has the solution `(r / d + (n0^(1-p) - r / d) exp(-d (1-p) t))^(1/(1-p))`, which
# tends to the equilibrium `(r / d)^(1/(1-p))`. The simulated volume should lie on it.

times, volume = trajectory.times, trajectory.get_values('tumour.x')
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
