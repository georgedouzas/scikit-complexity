"""
# Dynamic multi-species metabolic model

Two strains of one network compete for one pool of glucose, each with its own program, uptake kinetics, and biomass. The
strain that takes up glucose faster grows faster, so its share rises for as long as the glucose lasts.
"""

# %%
# The model is created by name from the catalog, which needs the extra `metabolism`. The primary view is the series of
# the biomass of the strains and the glucose.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import CoSimulationSolver, RK4Solver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'dynamic_multispecies_fba'
FRAME = 0.05

blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin='culture', name='culture'), collapse_panels=True)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
solver = CoSimulationSolver(sync_dt=FRAME, solvers={'culture': RK4Solver(dt=0.005)})
simulation = Simulation(create(MODEL), solver=solver, dt=FRAME, observers=[observer]).prepare().run(until=8.0)
trajectory = simulation.trajectory_

# %%
# The share of the faster strain
# ------------------------------
# The share should rise from one half while glucose lasts, and then stay, since the strains grow no more.

times = trajectory.times
fast = trajectory.values('culture/biomass_fast.x')
slow = trajectory.values('culture/biomass_slow.x')
glucose = trajectory.values('culture/glucose.c')
share = fast / (fast + slow)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(times, fast, label='fast strain')
axes[0].plot(times, slow, label='slow strain')
axes[0].set_ylabel('biomass (gDW/L)')
axes[0].legend()
axes[1].plot(times, share, label='share of the fast strain')
axes[1].plot(times, glucose / glucose[0], label='glucose, share of the start')
axes[1].set_xlabel('time (h)')
axes[1].legend()
fig.suptitle(f'The share of the fast strain rises from {share[0]:.2f} to {share[-1]:.2f}')
fig.tight_layout()
plt.show()
