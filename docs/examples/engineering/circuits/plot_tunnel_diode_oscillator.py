"""
# Tunnel-diode oscillator

The negative resistance of the tunnel diode amplifies small voltages and damps large ones, so a tiny initial voltage
grows into a self-sustained oscillation of fixed amplitude, the limit cycle of the van der Pol oscillator.
"""

# %%
# The model is created by name from the catalog with `L = C = 1`, `alpha = gamma = 1`, and the initial voltage 0.1, and
# runs for 50 time units. The default blueprint shows the voltage and the current in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tunnel_diode_oscillator'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=50)

# %%
# One limit cycle
# ---------------
# Runs started inside and outside should all converge on one closed orbit, whose voltage amplitude is close to
# `2 sqrt(alpha / (3 gamma)) = 1.15` for a weak nonlinearity and grows towards 2 as `alpha` becomes large.

fig, ax = plt.subplots(figsize=(6, 5))
for V0, i0 in [(0.1, 0.0), (2.0, 1.0), (-1.5, -1.5), (0.0, 2.5)]:
    run = Simulation(create(MODEL, V=V0, i=i0), dt=0.05).run(until=50).trajectory_
    ax.plot(run['circuit.V'], run['circuit.i'], linewidth=0.8)
    ax.plot(V0, i0, 'o', color='gray', markersize=4)
trajectory = simulation.trajectory_
amplitude = np.abs(trajectory['circuit.V'][trajectory.times > 30]).max()
for side in (-1, 1):
    ax.axvline(side * 2 * np.sqrt(1 / 3), color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('voltage V')
ax.set_ylabel('current i')
ax.set_title(f'Limit cycle of amplitude {amplitude:.2f}')
plt.show()
