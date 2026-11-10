"""
# Coupled Colpitts oscillators

Two chaotic Colpitts oscillators with natural frequencies 1.02 and 0.98 are coupled through the difference of their
second variables. Without coupling their phases drift apart, and for a moderate coupling the phases lock while the
amplitudes stay irregular.
"""

# %%
# The example builds `CoupledColpittsSystem` with the coupling `k = 0.4`. The default blueprint shows the three
# variables of both oscillators.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.circuits import CoupledColpittsSystem

SYSTEM = 'coupled_colpitts'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
trajectory = simulate(CoupledColpittsSystem(), observers=[observer])

# %%
# Phase difference
# ----------------
# The phase of an oscillator is the angle of its orbit in the plane of `y` and `z`, around their means, since the
# projection on `x` and `y` does not rotate around a single point. The phase difference should drift without coupling
# and stay within a turn for `k = 0.4`.

fig, ax = plt.subplots(figsize=(6, 4))
for coupling in [0.0, 0.4]:
    trajectory = simulate(CoupledColpittsSystem(k=coupling), until=400)
    settled = trajectory.times > 100
    phases = []
    for index in range(2):
        y = np.ravel(trajectory.get_values(f'oscillator_{index}.y'))[settled]
        z = np.ravel(trajectory.get_values(f'oscillator_{index}.z'))[settled]
        phases.append(np.unwrap(np.arctan2(z - z.mean(), y - y.mean())))
    ax.plot(trajectory.times[settled], (phases[0] - phases[1]) / (2 * np.pi), label=f'k = {coupling}')
ax.set_xlabel('time')
ax.set_ylabel('phase difference (turns)')
ax.set_title('Phase locking of two Colpitts oscillators')
ax.legend()
plt.show()
