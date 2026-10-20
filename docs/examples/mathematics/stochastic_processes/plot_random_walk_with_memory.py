"""
# Random walk with memory

Three hundred walkers move by a velocity that keeps half of its last value and gains a normal kick at every step, so
their paths are smoother than those of a simple random walk and spread out faster.
"""

# %%
# The walkers start at rest at 0 with the memory `alpha = 0.5` and kicks of deviation 1, and walk for 200 steps. The
# blueprint shows the positions of all walkers over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'random_walk_with_memory'

system = create(MODEL, n_walkers=300, alpha=0.5, sigma=1.0)
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/walkers/x',
        name='positions x',
        axis_x=rrb.TimeAxis(view_range=whole_run),
        plot_legend=rrb.PlotLegend(visible=False),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=200)

# %%
# Mean squared displacement
# -------------------------
# The second moments of `(x, v)` follow a linear recursion, which gives the exact mean squared displacement. It should
# match the simulation and grow at long times with the slope `sigma^2 / (1 - alpha)^2 = 4`, larger than the slope
# `sigma^2 / (1 - alpha^2)` that steps of the same variance would give without correlation.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
moments, exact = np.zeros(3), []
for _ in times:
    exact.append(moments[0])
    xx, xv, vv = moments
    moments = np.array([xx + 2 * xv + vv, 0.5 * (xv + vv), 0.25 * vv + 1.0])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(times, np.mean(trajectory['walkers.x'] ** 2, axis=1), label='simulation')
ax.plot(times, exact, '--', color='black', label='exact second moment')
ax.plot(times, times / (1 - 0.5**2), ':', color='grey', label='uncorrelated steps')
ax.set_xlabel('step t')
ax.set_ylabel('mean squared displacement')
ax.set_title('Correlated velocities speed up diffusion')
ax.legend()
fig.tight_layout()
plt.show()
