"""
# Lighthill-Whitham-Richards traffic

Free-flowing traffic runs into the back of a queue. The boundary between them is a shock, a sudden jump of density
that moves upstream, against the traffic, at a constant speed, as arriving cars join the queue.
"""

# %%
# The model is created by name from the catalog on a road of 200 cells with free traffic at density 0.3 running into a
# queue at density 0.8, with the Greenshields speed law, and is integrated with SSPRK for one time unit. The default
# blueprint puts the history of the density first, where the shock is the edge that moves left.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import SSPRK
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'lighthill_whitham_richards'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.02, solver=SSPRK(), observers=[observer]).run(until=1.0)

# %%
# Shock speed
# -----------
# By the Rankine-Hugoniot condition, the shock between the densities `rho_u = 0.3` and `rho_d = 0.8` should move at
# `vmax (1 - (rho_u + rho_d) / rho_max) = -0.1`, so its position should fall on a line of that slope from the middle of
# the road.

trajectory = simulation.trajectory_
rho = fields_of(trajectory, 'road.rho')
x = (np.arange(rho.shape[1]) + 0.5) / rho.shape[1]
fronts = [x[np.argmax(row > 0.55)] for row in rho]
fig, (profiles, front) = plt.subplots(1, 2, figsize=(10, 4))
for index in range(0, len(rho), 10):
    profiles.plot(x, rho[index], label=f't = {trajectory.times[index]:.1f}')
profiles.set_xlabel('position')
profiles.set_ylabel('density')
profiles.set_title('The queue grows upstream')
profiles.legend()
front.plot(trajectory.times, fronts, 'o', markersize=3, label='simulated shock')
front.plot(trajectory.times, 0.5 - 0.1 * trajectory.times, color='black', label='Rankine-Hugoniot speed -0.1')
front.set_xlabel('time')
front.set_ylabel('position of the shock')
front.set_title('The shock moves at the predicted speed')
front.legend()
fig.tight_layout()
plt.show()
