"""
# Power-law models of biochemical systems theory

An unbranched pathway whose end product inhibits its supply, written as an S-system, relaxes to the steady state that
the linear equations in the logarithms of the concentrations predict.
"""

# %%
# The model is created by name from the catalog. The pathway `X1 -> X2 -> X3` has power-law fluxes, and `X3` inhibits
# the supply of `X1` with a kinetic order of -0.5. The observer saves every frame of the 20 time units with the
# default blueprint, whose primary view is the time series of the three concentrations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'biochemical_systems_theory'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# The steady state of an S-system
# -------------------------------
# At a steady state of `dx_i/dt = alpha_i prod_j x_j^g_ij - beta_i prod_j x_j^h_ij` the production equals the
# degradation, which is linear in `y = log x`: `(g - h) y = log(beta / alpha)`. The concentrations should approach the
# solution, `x = (2, 2, 2)`. The same pathway written in the generalized mass action form, four fluxes with the same
# rates, should give the same curves.

g = np.array([[0.0, 0.0, -0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
h = 0.5 * np.eye(3)
steady = np.exp(np.linalg.solve(g - h, np.log(np.array([1.0, 1.0, 1.0]) / np.array([2.0, 1.0, 1.0]))))
gma = Simulation(create(MODEL, form='gma'), dt=0.1).run(until=20).trajectory_
trajectory = simulation.trajectory_
fig, ax = plt.subplots(figsize=(7, 4))
for index, color in zip((1, 2, 3), ('C0', 'C1', 'C2'), strict=False):
    ax.plot(trajectory.times, trajectory[f'X{index}.x'], color=color, label=f'X{index}, S-system')
    ax.plot(gma.times, gma[f'X{index}.x'], ':', color='black')
    ax.axhline(steady[index - 1], color=color, linestyle='--', alpha=0.5)
ax.plot([], [], ':', color='black', label='generalized mass action')
ax.set_xlabel('time')
ax.set_ylabel('concentration')
ax.set_title('Both forms relax to the log-linear steady state')
ax.legend()
fig.tight_layout()
plt.show()
