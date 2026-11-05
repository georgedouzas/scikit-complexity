"""
# Regulatory flux balance analysis

Flux balance alone would use glucose and acetate at once. Boolean gene rules alternate with the program, so glucose
represses the genes of acetate kinase and the culture uses its substrates in turn. Between them it waits for the rules
to release the genes, the lag of a diauxic shift.
"""

# %%
# The system is created by name from the catalog, which needs the extra `metabolism`. The Boolean nodes and the program
# run in a discrete partition beside the differential equations of the culture. The primary view is the series of the
# biomass, the glucose, and the acetate.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import CoSimulationSolver, RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'regulatory_fba'
FRAME = 0.05

blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin='culture', name='culture'), collapse_panels=True)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=2)
solver = CoSimulationSolver(sync_dt=FRAME, solvers={'culture': RK4Solver(dt=0.005)})
simulation = Simulation(create_system(SYSTEM), solver=solver, dt=FRAME, observers=[observer]).prepare().run(until=8.0)
trajectory = simulation.trajectory_

# %%
# Two phases with a lag
# ---------------------
# The acetate uptake should stay at zero while glucose lasts, and rise only after the delay of the Boolean chain, while
# the biomass stands still.

times = trajectory.times
biomass = trajectory.values('culture/biomass.x')
glucose = trajectory.values('culture/glucose.c')
acetate = trajectory.values('culture/acetate.c')
uptake = -trajectory.values('control/cell.v_EX_ac_e')
depleted = times[np.argmax(glucose <= 1e-3)]
released = times[np.argmax(uptake > 1e-6)]
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(times, glucose, label='glucose')
axes[0].plot(times, acetate, label='acetate')
axes[0].set_ylabel('concentration (mM)')
axes[0].legend()
axes[1].semilogy(times, biomass, label='biomass')
axes[1].axvspan(depleted, released, color='tab:gray', alpha=0.3, label='lag')
axes[1].set_xlabel('time (h)')
axes[1].set_ylabel('biomass (gDW/L)')
axes[1].legend()
fig.suptitle(f'Glucose is gone at {depleted:.2f} h, and acetate uptake starts at {released:.2f} h')
fig.tight_layout()
plt.show()
