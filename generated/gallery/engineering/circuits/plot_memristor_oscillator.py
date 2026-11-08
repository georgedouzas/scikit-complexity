"""
# Memristor oscillator

A capacitor, an inductor, and a memristor in series are enough for chaos: the memristance changes with the current
that has flowed through it, and the voltage oscillates with irregular, never-repeating amplitudes.
"""

# %%
# The system is built from its class with `C = 1`, `L = 3`, `alpha = 0.6`, and `beta = 1.5`, and runs for
# 200 time units. The default blueprint shows the voltage, the current, and the memristor state in one time series
# view.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import lyapunov
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.circuits import MemristorOscillatorSystem

SYSTEM = 'memristor_oscillator'

system = MemristorOscillatorSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=8)
trajectory = simulate(system, observers=[observer])

# %%
# A chaotic attractor
# -------------------
# The trajectory should fill a band in the voltage-current plane instead of closing on a cycle, and the largest
# Lyapunov exponent of the flow should be positive.

settled = trajectory.times > 20
exponent = lyapunov(system, n_exponents=1, until=500.0, dt=0.01).exponents[0]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(trajectory.values('circuit.v')[settled], trajectory.values('circuit.i')[settled], linewidth=0.5)
ax.set_xlabel('capacitor voltage v')
ax.set_ylabel('current i')
ax.set_title(f'Chaotic attractor, largest Lyapunov exponent {exponent:.3f}')
plt.show()
