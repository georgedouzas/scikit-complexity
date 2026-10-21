"""
# Chua's circuit

The voltage across the first capacitor of Chua's circuit spirals around one of two outer equilibria, then jumps to the
other at irregular times: the double-scroll chaotic attractor.
"""

# %%
# The model is created by name from the catalog with the piecewise-linear diode and its default parameters, and runs
# for 100 time units. The default blueprint shows the two capacitor voltages `x` and `y` and the inductor current `z` in
# one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'chua'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=100)

# %%
# The double scroll
# -----------------
# Projected on the two capacitor voltages, the trajectory should wind around both outer equilibria in two scrolls, and
# the largest Lyapunov exponent of the flow should be positive.

trajectory = simulation.trajectory_
settled = trajectory.times > 10
exponents = lyapunov_spectrum(system, n=1, until=300.0, dt=0.01)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(trajectory['circuit.x'][settled], trajectory['circuit.z'][settled], linewidth=0.5)
ax.set_xlabel('voltage x')
ax.set_ylabel('current z')
ax.set_title(f'Double scroll, largest Lyapunov exponent {exponents[0]:.2f}')
plt.show()
