"""
# Chua's circuit

The voltage across the first capacitor of Chua's circuit spirals around one of two outer equilibria, then jumps to the
other at irregular times: the double-scroll chaotic attractor.
"""

# %%
# The system is built from its class with the piecewise-linear diode and its default parameters, and runs
# for 100 time units. The default blueprint shows the two capacitor voltages `x` and `y` and the inductor current `z` in
# one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.circuits import ChuaSystem

SYSTEM = 'chua'

system = ChuaSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=8)
trajectory = simulate(system, observers=[observer])

# %%
# The double scroll
# -----------------
# Projected on the two capacitor voltages, the trajectory should wind around both outer equilibria in two scrolls, and
# the largest Lyapunov exponent of the flow should be positive.

settled = trajectory.times > 10
exponents = lyapunov(system, n_exponents=1, until=300.0, dt=0.01).exponents
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(trajectory.get_values('circuit.x')[settled], trajectory.get_values('circuit.z')[settled], linewidth=0.5)
ax.set_xlabel('voltage x')
ax.set_ylabel('current z')
ax.set_title(f'Double scroll, largest Lyapunov exponent {exponents[0]:.2f}')
plt.show()
