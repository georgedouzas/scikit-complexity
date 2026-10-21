"""
# Memristor oscillator

A capacitor, an inductor, and a memristor in series are enough for chaos: the memristance changes with the current
that has flowed through it, and the voltage oscillates with irregular, never-repeating amplitudes.
"""

# %%
# The model is created by name from the catalog with `C = 1`, `L = 3`, `alpha = 0.6`, and `beta = 1.5`, and runs for
# 200 time units. The default blueprint shows the voltage, the current, and the memristor state in one time series
# view.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'memristor_oscillator'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=200)

# %%
# A chaotic attractor
# -------------------
# The trajectory should fill a band in the voltage-current plane instead of closing on a cycle, and the largest
# Lyapunov exponent of the flow should be positive.

trajectory = simulation.trajectory_
settled = trajectory.times > 20
exponent = lyapunov_spectrum(system, n=1, until=500.0, dt=0.01)[0]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(trajectory['circuit.v'][settled], trajectory['circuit.i'][settled], linewidth=0.5)
ax.set_xlabel('capacitor voltage v')
ax.set_ylabel('current i')
ax.set_title(f'Chaotic attractor, largest Lyapunov exponent {exponent:.3f}')
plt.show()
