"""
# Eigen's quasispecies

Mutations carry a population away from the master sequence and selection pulls it back. Below the error threshold the
two balance in a quasispecies, a stable cloud of mutants around a master sequence that keeps a share `1 - u / sigma`;
above it the master sequence is lost, the error catastrophe.
"""

# %%
# The model is created by name from the catalog with the mutation rate `u = 0.3` below the selective disadvantage
# `sigma = 0.5`, starting from the master sequence alone. The blueprint shows the shares of the master sequence and
# of the first five classes of mutants in one time series view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'eigen_quasispecies'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
shares = rrb.TimeSeriesView(
    origin='/',
    contents=['/master/**'] + [f'/k{k}/**' for k in range(1, 6)],
    name='shares of the classes',
    defaults=[rr.SeriesLines.from_fields(widths=2.5)],
    axis_x=rrb.TimeAxis(view_range=whole_run),
)
blueprint = rrb.Blueprint(
    shares,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, u=0.3, sigma=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=40)

# %%
# The error threshold
# -------------------
# Sweeping the mutation rate, the final share of the master sequence should follow `1 - u / sigma` and vanish at the
# error threshold `u = sigma`.

rates = np.linspace(0.0, 0.7, 15)
shares = [
    Simulation(create(MODEL, u=rate, sigma=0.5), dt=1).run(until=200).trajectory_['master.x'][-1] for rate in rates
]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rates, shares, 'o', label='simulated')
ax.plot(rates, np.clip(1 - rates / 0.5, 0, None), '--', color='black', label='1 - u / sigma')
ax.axvline(0.5, color='grey', linestyle=':', label='error threshold')
ax.set_xlabel('mutation rate u')
ax.set_ylabel('share of the master sequence')
ax.set_title('The master sequence is lost at the error threshold')
ax.legend()
plt.show()
