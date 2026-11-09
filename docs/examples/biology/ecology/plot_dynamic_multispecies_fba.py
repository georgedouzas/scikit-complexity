"""
# Dynamic multi-species metabolic system

Two strains of one network compete for one pool of glucose, each with its own program, uptake kinetics, and biomass. The
strain that takes up glucose faster grows faster, so its share rises for as long as the glucose lasts.
"""

# %%
# The system is built from its class, which needs the extra `metabolism`. The primary view is the series of the
# biomass of the strains and the glucose.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import CoSimulationSolver, RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import DynamicMultispeciesFBASystem

SYSTEM = 'dynamic_multispecies_fba'
FRAME = 0.05

blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin='culture', name='culture'), collapse_panels=True)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
solver = CoSimulationSolver(sync_dt=FRAME, solvers={'culture': RK4Solver(dt=0.005)})
trajectory = simulate(DynamicMultispeciesFBASystem(), solver=solver, dt=FRAME, observers=[observer])

# %%
# The share of the faster strain
# ------------------------------
# The share should rise from one half while glucose lasts, and then stay, since the strains grow no more.

times = trajectory.times
fast = trajectory.get_values('culture/biomass_fast.x')
slow = trajectory.get_values('culture/biomass_slow.x')
glucose = trajectory.get_values('culture/glucose.c')
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
