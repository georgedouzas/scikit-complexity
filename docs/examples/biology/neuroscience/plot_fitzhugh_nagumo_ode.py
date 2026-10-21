"""
# FitzHugh-Nagumo neuron

Under a constant current the FitzHugh-Nagumo neuron fires periodically: the fast potential jumps up, the slow recovery
catches up and pulls it down, and the cycle repeats as relaxation oscillations.
"""

# %%
# The model is created by name from the catalog with the current `I = 0.5`, inside the range of periodic firing. The
# observer saves every second frame with the default blueprint, whose primary view is the time series of the
# potential and the recovery.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fitzhugh_nagumo_ode'

system = create(MODEL, I=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=100)

# %%
# Limit cycle and nullclines
# --------------------------
# The potential stops changing on the cubic `w = v - v³/3 + I` and the recovery on the line `w = (v + a) / b`. The
# fixed point where they cross lies on the middle branch of the cubic and is unstable, so the trajectory should
# settle on a cycle that hugs the outer branches of the cubic.

v, w = simulation.trajectory_['neuron.v'], simulation.trajectory_['neuron.w']
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
