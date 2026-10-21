"""
# Payne traffic model

A bump of density on a ring road of uniform traffic splits and travels, while the relaxation of the speed towards the
Greenshields law and the anticipation of drivers smooth it out, so the road returns towards uniform flow.
"""

# %%
# The model is created by name from the catalog on a ring of 200 cells at density 0.3 with a bump of 0.2, the
# relaxation time `tau = 0.5`, and is integrated with SSPRK for one time unit. The default blueprint puts the history of
# the density first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import SSPRK
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'payne'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.01, solver=SSPRK(), observers=[observer]).run(until=1.0)

# %%
# Conservation and smoothing
# --------------------------
# The density obeys a conservation law on a ring, so the number of cars should stay exactly constant, while the peak of
# the bump should fall as it spreads.

trajectory = simulation.trajectory_
rho = fields_of(trajectory, 'road.rho')
x = (np.arange(rho.shape[1]) + 0.5) / rho.shape[1]
cars = rho.sum(axis=1) / rho.shape[1]
fig, (profiles, totals) = plt.subplots(1, 2, figsize=(10, 4))
for index in range(0, len(rho), 25):
    profiles.plot(x, rho[index], label=f't = {trajectory.times[index]:.2f}')
profiles.set_xlabel('position on the ring')
profiles.set_ylabel('density')
profiles.set_title('The bump travels and spreads')
profiles.legend()
totals.plot(trajectory.times, (cars - cars[0]) / cars[0], label='relative change of the number of cars')
totals.plot(trajectory.times, rho.max(axis=1) - 0.3, label='height of the bump above 0.3')
totals.set_xlabel('time')
totals.set_ylabel('value')
totals.set_title('Cars are conserved while the bump decays')
totals.legend()
fig.tight_layout()
plt.show()
