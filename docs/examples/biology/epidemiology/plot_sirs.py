"""
# SIRS

When immunity wanes, the recovered return to the susceptibles and the disease comes back. After a first large
outbreak, ever smaller waves follow until the infection settles at an endemic level.
"""

# %%
# The model is created by name from the catalog with its default rates, an immunity that lasts 50 days on average,
# and run for 300 days. The observer saves every second day with the default blueprint, whose primary view is the
# time series of the three compartments.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sirs'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=300)

# %%
# Damped waves to the endemic equilibrium
# ---------------------------------------
# The endemic equilibrium has `S = gamma / beta` susceptibles and `I = xi (N - S) / (gamma + xi)` infected. In the
# `(S, I)` plane the trajectory should spiral into that point.

trajectory = simulation.trajectory_
susceptible, infected = trajectory['S.n'], trajectory['I.n']
beta, gamma, xi, total = 0.0003, 0.1, 0.02, 1000.0
s_star = gamma / beta
i_star = xi * (total - s_star) / (gamma + xi)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(susceptible, infected, label='trajectory')
ax.plot(s_star, i_star, 'o', color='black', label=f'endemic equilibrium ({s_star:.0f}, {i_star:.0f})')
ax.set_xlabel('susceptible')
ax.set_ylabel('infected')
ax.set_title(f'Final state ({susceptible[-1]:.0f}, {infected[-1]:.0f})')
ax.legend()
plt.show()
