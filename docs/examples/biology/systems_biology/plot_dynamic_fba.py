"""
# Dynamic flux balance analysis

A batch culture changes its medium slowly compared with its metabolism, so the fluxes can follow an optimum at every
moment. The static optimisation approach solves the program, holds the fluxes over an interval, and integrates the
biomass and the glucose, until the glucose runs out and growth stops.
"""

# %%
# The system is created by name from the catalog, which needs the extra `metabolism`, with a program in a discrete
# partition that solves at every synchronisation point. The primary view is the series of the biomass and the glucose.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import CoSimulationSolver, RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'dynamic_fba'
FRAME = 0.05
KM = 0.015

blueprint = rrb.Blueprint(rrb.TimeSeriesView(origin='culture', name='culture'), collapse_panels=True)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
solver = CoSimulationSolver(sync_dt=FRAME, solvers={'culture': RK4Solver(dt=0.005)})
simulation = Simulation(create_system(SYSTEM), solver=solver, dt=FRAME, observers=[observer]).prepare().run(until=10.0)
trajectory = simulation.trajectory_

# %%
# One exponential phase and a plateau
# -----------------------------------
# While glucose exceeds ten half-saturation constants, the logarithm of the biomass should rise with the slope of the
# optimum, 0.874 per hour. After depletion the biomass should stay at the yield of the glucose.

times = trajectory.times
biomass = trajectory.values('culture/biomass.x')
glucose = trajectory.values('culture/glucose.c')
plenty = glucose > 10 * KM
slope, intercept = np.polyfit(times[plenty], np.log(biomass[plenty]), 1)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(times, np.log(biomass), label='biomass')
axes[0].plot(times[plenty], slope * times[plenty] + intercept, '--', label=f'fit, slope {slope:.3f} 1/h')
axes[0].set_ylabel('log biomass')
axes[0].legend()
axes[1].plot(times, glucose)
axes[1].set_xlabel('time (h)')
axes[1].set_ylabel('glucose (mM)')
fig.suptitle(f'Slope {slope:.3f} 1/h, and a plateau at {biomass[-1]:.3f} gDW/L after the glucose runs out')
fig.tight_layout()
plt.show()
