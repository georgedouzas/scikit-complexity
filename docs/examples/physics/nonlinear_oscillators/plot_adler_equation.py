"""
# Adler equation

An oscillator weakly driven at a slightly different frequency locks to the drive: the phase difference stops at a fixed
value when the detuning is smaller than the coupling, and slips by full turns at a steady rate when it is larger.
"""

# %%
# The detuning `Δω = 0.5` is half the coupling `ε = 1`, so the phase difference starts at 0 and settles at the locked
# value. The default blueprint shows the phase difference as a time series.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'adler_equation'

system = create(MODEL, phi=0.0, detuning=0.5, epsilon=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# Locking and phase slips
# -----------------------
# For `|Δω| < ε` the phase difference should lock, so its mean rate of change is zero. Above the locking threshold it
# drifts at the mean rate `sqrt(Δω^2 - ε^2)`, the square-root law of the saddle-node on a circle. The measured
# rates over a sweep of the detuning should follow that curve.

detunings = np.linspace(0.0, 3.0, 25)
rates = []
for detuning in detunings:
    run = Simulation(create(MODEL, phi=0.0, detuning=detuning, epsilon=1.0), dt=0.05).run(until=400)
    phase = np.unwrap(np.ravel(run.trajectory_['difference.phi']))
    times = np.asarray(run.trajectory_.times)
    half = times > 200
    rates.append(np.polyfit(times[half], phase[half], 1)[0])
grid = np.linspace(0.0, 3.0, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, np.sqrt(np.clip(grid**2 - 1, 0, None)), color='black', label='sqrt(Δω² - ε²)')
ax.plot(detunings, rates, 'o', label='simulated')
ax.axvspan(0, 1, color='grey', alpha=0.2, label='locked')
ax.set_xlabel('detuning Δω')
ax.set_ylabel('mean rate of the phase difference')
ax.set_title('Phase locking below ε = 1')
ax.legend()
plt.show()
