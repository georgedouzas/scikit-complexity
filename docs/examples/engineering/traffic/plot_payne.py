"""
# Payne traffic system

A bump of density on a ring road of uniform traffic splits and travels, while the relaxation of the speed towards the
Greenshields law and the anticipation of drivers smooth it out, so the road returns towards uniform flow.
"""

# %%
# The example builds `PayneSystem` on a ring of 200 cells at density 0.3 with a bump of 0.2, the
# relaxation time `tau = 0.5`, and is integrated with SSPRK for one time unit. The default blueprint puts the history of
# the density first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import SSPRKSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.traffic import PayneSystem

SYSTEM = 'payne'

system = PayneSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, dt=0.01, solver=SSPRKSolver(), observers=[observer])

# %%
# Conservation and smoothing
# --------------------------
# The density obeys a conservation law on a ring, so the number of cars should stay exactly constant, while the peak of
# the bump should fall as it spreads.

rho = trajectory.get_lattice('road.rho')
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
