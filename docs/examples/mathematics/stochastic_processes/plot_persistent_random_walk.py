"""
# Persistent random walk

Three hundred walkers keep the direction of their last step with probability 0.8, so they run in long straight
stretches and fan out faster than walkers that choose every step afresh.
"""

# %%
# The walkers start at 0 with random directions and walk for 200 steps. The blueprint shows the positions of all
# walkers over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'persistent_random_walk'

system = create(MODEL, n_walkers=300, persistence=0.8)
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
# Successive steps are correlated with `rho = 2 q - 1 = 0.6`, so the mean squared displacement should grow as
# `t (1 + rho) / (1 - rho) - 2 rho (1 - rho^t) / (1 - rho)^2`, four times faster at long times than the `t` of a
# simple random walk, which the dotted line shows.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
rho = 2 * 0.8 - 1
exact = times * (1 + rho) / (1 - rho) - 2 * rho * (1 - rho**times) / (1 - rho) ** 2
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(times, np.mean(trajectory['walkers.x'] ** 2, axis=1), label='simulation')
ax.plot(times, exact, '--', color='black', label='persistent walk')
ax.plot(times, times, ':', color='grey', label='simple random walk')
ax.set_xlabel('step t')
ax.set_ylabel('mean squared displacement')
ax.set_title('Persistence speeds up diffusion fourfold')
ax.legend()
fig.tight_layout()
plt.show()
