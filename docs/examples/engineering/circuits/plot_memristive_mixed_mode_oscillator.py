"""
# Memristive mixed-mode oscillator

A circuit with a cubic flux-controlled memristor and a small first capacitance runs on two time scales. Its memristor
voltage settles on a mixed-mode oscillation: two large spikes, then eleven small oscillations that grow slowly until the
next pair of spikes, the pattern `2^11` that Trzaska reports.
"""

# %%
# The system is built from its class with Trzaska's parameters, `epsilon = 0.01`, `R = 2`,
# `alpha = 1.55`, `beta = 0.15`, `e = -0.04`, `a = -0.34`, and `b = 0.05`, and starts at rest. The default blueprint
# shows the four states in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.circuits import MemristiveMixedModeOscillatorSystem

SYSTEM = 'memristive_mixed_mode_oscillator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
trajectory = simulate(MemristiveMixedModeOscillatorSystem(), observers=[observer])

# %%
# Large and small oscillations
# ----------------------------
# After a transient the voltage of the memristor should repeat with a period near 7.8: two spikes far above the rest,
# followed by eleven small peaks.

settled = trajectory.times > 100
x = np.ravel(trajectory.values('circuit.x'))[settled]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trajectory.times[settled], x, linewidth=0.8)
ax.set_xlabel('time')
ax.set_ylabel('memristor voltage x')
ax.set_title('Mixed-mode oscillation 2^11')
plt.show()
