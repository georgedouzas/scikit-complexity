"""
# FitzHugh-Nagumo neuron

Under a constant current the FitzHugh-Nagumo neuron fires periodically: the fast potential jumps up, the slow recovery
catches up and pulls it down, and the cycle repeats as relaxation oscillations.
"""

# %%
# The system is built from its class with the current `I = 0.5`, inside the range of periodic firing. The observer
# saves every second frame with the default blueprint, whose primary view is the time series of the potential and the
# recovery.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import FitzhughNagumoODESystem

SYSTEM = 'fitzhugh_nagumo_ode'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(FitzhughNagumoODESystem(I=0.5), observers=[observer])

# %%
# Limit cycle and nullclines
# --------------------------
# The potential stops changing on the cubic `w = v - v³/3 + I` and the recovery on the line `w = (v + a) / b`. The
# fixed point where they cross lies on the middle branch of the cubic and is unstable, so the trajectory should
# settle on a cycle that hugs the outer branches of the cubic.

v, w = trajectory.values('neuron.v'), trajectory.values('neuron.w')
a, b, current = 0.7, 0.8, 0.5
grid = np.linspace(-2.5, 2.5, 200)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(grid, grid - grid**3 / 3 + current, '--', color='tab:blue', label="v' = 0")
ax.plot(grid, (grid + a) / b, '--', color='tab:orange', label="w' = 0")
ax.plot(v, w, color='black', label='trajectory')
ax.set_ylim(-1, 2.5)
ax.set_xlabel('potential v')
ax.set_ylabel('recovery w')
ax.set_title('Relaxation oscillations around an unstable fixed point')
ax.legend()
plt.show()
