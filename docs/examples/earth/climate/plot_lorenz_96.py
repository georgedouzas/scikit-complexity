"""
# Lorenz-96

A small kick to one site of the ring spreads to all twenty sites, and the steady state breaks into chaotic waves that
travel around the latitude circle, from site to site.
"""

# %%
# The model is created by name from the catalog with 20 sites and the forcing `F = 8`, with the first site displaced by
# 0.01 from the steady state, and runs for 30 time units. The blueprint shows three sites of the ring in one time series
# view, hiding the other seventeen.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'lorenz_96'

shown = [site in (0, 7, 14) for site in range(20)]
sites = rrb.TimeSeriesView(
    origin='/ring/x',
    name='sites 0, 7, and 14',
    overrides={'ring/x': rr.SeriesLines.from_fields(visible_series=shown, widths=2.0)},
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
    plot_legend=rrb.PlotLegend(visible=False),
)
blueprint = rrb.Blueprint(
    sites,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=3)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=30)

# %%
# Space-time diagram of the waves
# -------------------------------
# Drawn against site and time, the values should show the perturbation spreading from site 0 to the whole ring, then
# tilted stripes of travelling waves. The lagged correlation between neighbouring sites should peak away from zero lag,
# at the time a wave takes to travel from one site to the next.

trajectory = simulation.trajectory_
x = trajectory['ring.x']
times = trajectory.times
late = x[times > 10] - x[times > 10].mean()
lags = np.arange(-20, 21)
correlation = [
    np.mean([np.corrcoef(np.roll(late[:, i], lag), late[:, (i + 1) % 20])[0, 1] for i in range(20)]) for lag in lags
]
fig, (diagram, travel) = plt.subplots(1, 2, figsize=(10, 4))
image = diagram.imshow(x.T, aspect='auto', origin='lower', extent=(0, times[-1], -0.5, 19.5), cmap='RdBu_r')
fig.colorbar(image, ax=diagram, label='x')
diagram.set_xlabel('time')
diagram.set_ylabel('site')
diagram.set_title('Spreading and travelling waves')
travel.plot(lags * 0.05, correlation, 'o-', markersize=3)
travel.axvline(0.0, color='gray', linewidth=0.8)
travel.set_xlabel('time lag between site i and site i + 1')
travel.set_ylabel('correlation')
travel.set_title('Neighbouring sites repeat each other with a lag')
fig.tight_layout()
plt.show()
