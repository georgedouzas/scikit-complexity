"""
# Sutera-Fraedrich tipping

The global mean temperature of a zero-dimensional energy balance fluctuates around its warm equilibrium near 291.6 K
under the noise of the weather. The warm state is stable, yet a run of fluctuations carries the temperature over the
unstable equilibrium near 281 K, and the climate tips: noise-induced tipping. The view shows the temperature
fluctuating at its warm level and then falling abruptly to the cold state.
"""

# %%
# The model is created by name from the catalog for noise-induced tipping: the controls of Table 1 of Fang et al.
# held fixed and noise of variance 1 K^2 per year. The run lasts 300 years with steps of 0.05 years, and the observer
# saves every second year. The blueprint shows the temperature over the whole run.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'sutera_fraedrich_tipping'

solver = describe(MODEL).defaults['solver']
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/earth/T', name='temperature (K)', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, tipping='noise')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(system, dt=0.5, solver=solver, observers=[observer], random_state=0).run(until=300)

# %%
# Three types of tipping
# ----------------------
# In bifurcation-induced tipping, the control `mu` falls at `4e-4` per year, and the warm state should track its
# equilibrium until the equilibrium meets the unstable one at the fold `mu = 4 a2 / (c1 b2^2) = 0.76`, then fall. In
# rate-induced tipping, the controls ramp to values that keep a warm equilibrium, and a slow ramp at `rho = 0.16`
# should follow it to 303 K while a fast ramp at `rho = 0.2` should lose it and tip. Fang et al. put the critical
# rate near 0.175.

bifurcation = Simulation(create(MODEL, tipping='bifurcation'), dt=1.0, solver=solver).run(until=720).trajectory_
fold = 4 * 0.2 / (9.7137e9 * 1.04e-5**2)
fig, (ramped, rates) = plt.subplots(1, 2, figsize=(10, 4))
ramped.plot(bifurcation['earth.mu'], bifurcation['earth.T'])
ramped.axvline(fold, color='black', linestyle=':', label=f'fold at mu = {fold:.3f}')
ramped.set_xlim(1.0, 0.7)
ramped.set_ylim(150, 300)
ramped.set_xlabel('control mu, falling over time')
ramped.set_ylabel('temperature (K)')
ramped.set_title('B-tipping past the fold')
ramped.legend()
for rho in (0.16, 0.2):
    trajectory = Simulation(create(MODEL, tipping='rate', rho=rho), dt=0.5, solver=solver).run(until=150).trajectory_
    rates.plot(trajectory.times, trajectory['earth.T'], label=f'rho = {rho}')
rates.set_ylim(240, 310)
rates.set_xlabel('time (years)')
rates.set_ylabel('temperature (K)')
rates.set_title('R-tipping above a critical rate')
rates.legend()
fig.tight_layout()
plt.show()
