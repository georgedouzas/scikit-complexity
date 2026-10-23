"""
# Optimal immunotherapy

An oncologist combines adoptive cell therapy and IL-2 injections to minimise the tumour burden plus the cost of both
treatments. Both treatments are given at full strength while the tumour is large, the effector cells expand, and the
tumour is cleared, after which the treatments stop.
"""

# %%
# The model is created by name from the catalog, the Kirschner-Panetta model in units of `10^5` cells, with a tumour
# of low antigenicity that escapes the immune response when untreated. The Pontryagin solver of its catalog entry solves
# the problem on 1001 nodes over 50 days, fine enough for the fast decay of IL-2. The default blueprint shows the
# cells, IL-2, and both treatments in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'optimal_immunotherapy'

system = create(MODEL)
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=0.5,
    solver=entry.defaults['solver'],
    observers=[observer],
    observables=entry.observables_of(system),
).run(until=50)

# %%
# Treated against untreated
# -------------------------
# With prohibitive weights on both treatments, the optimal treatment is nearly none and the tumour grows at nearly its
# intrinsic rate 0.18 per day. With the optimal treatments the tumour should fall by orders of magnitude within weeks.

untreated = Simulation(create(MODEL, w1=1e6, w2=1e6), dt=0.5, solver=entry.defaults['solver']).run(until=50)
trajectory, free = simulation.trajectory_, untreated.trajectory_
fig, (tumour, doses) = plt.subplots(1, 2, figsize=(10, 4))
tumour.semilogy(free.times, free['site.y'], label='untreated')
tumour.semilogy(trajectory.times, trajectory['site.y'] + 1e-9, label='optimal treatment')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells y (10^5)')
tumour.set_title('The treatment clears the tumour')
tumour.legend()
doses.plot(trajectory.times, trajectory['site.u1'], label='adoptive therapy u1')
doses.plot(trajectory.times, trajectory['site.u2'], label='IL-2 injection u2')
doses.set_xlabel('time (days)')
doses.set_title('Treat hard while the tumour is large')
doses.legend()
fig.tight_layout()
plt.show()
