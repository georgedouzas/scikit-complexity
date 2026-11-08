"""
# Tunnel-diode oscillator

The negative resistance of the tunnel diode amplifies small voltages and damps large ones, so a tiny initial voltage
grows into a self-sustained oscillation of fixed amplitude, the limit cycle of the van der Pol oscillator.
"""

# %%
# The system is built from its class with `L = C = 1`, `alpha = gamma = 1`, and the initial voltage 0.1, and
# runs for 50 time units. The default blueprint shows the voltage and the current in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.circuits import TunnelDiodeOscillatorSystem

SYSTEM = 'tunnel_diode_oscillator'

system = TunnelDiodeOscillatorSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(system, observers=[observer])

# %%
# One limit cycle
# ---------------
# Runs started inside and outside should all converge on one closed orbit, whose voltage amplitude is close to
# `2 sqrt(alpha / (3 gamma)) = 1.15` for a weak nonlinearity and grows towards 2 as `alpha` becomes large.

fig, ax = plt.subplots(figsize=(6, 5))
for V0, i0 in [(0.1, 0.0), (2.0, 1.0), (-1.5, -1.5), (0.0, 2.5)]:
    run = simulate(TunnelDiodeOscillatorSystem(V=V0, i=i0))
    ax.plot(run.values('circuit.V'), run.values('circuit.i'), linewidth=0.8)
    ax.plot(V0, i0, 'o', color='gray', markersize=4)
amplitude = np.abs(trajectory.values('circuit.V')[trajectory.times > 30]).max()
for side in (-1, 1):
    ax.axvline(side * 2 * np.sqrt(1 / 3), color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('voltage V')
ax.set_ylabel('current i')
ax.set_title(f'Limit cycle of amplitude {amplitude:.2f}')
plt.show()
