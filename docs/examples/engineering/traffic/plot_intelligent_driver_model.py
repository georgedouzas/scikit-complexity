"""
# Intelligent driver model

Twenty cars start at rest on a ring road and accelerate together. Each keeps a safe time gap to the car ahead, so the
platoon settles at the one speed where the gap that the ring leaves to every car equals the desired gap at that
speed.
"""

# %%
# The model is created by name from the catalog with 20 cars of length 5 on a ring of 500 m, the desired speed 30 m/s,
# and the time gap 1.5 s, and runs for 100 s. The blueprint shows the speeds of all cars in one time series view,
# without a legend.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'intelligent_driver_model'

speeds = rrb.TimeSeriesView(
    origin='/cars/v',
    name='speeds',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
    plot_legend=rrb.PlotLegend(visible=False),
)
blueprint = rrb.Blueprint(
    speeds,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# The equilibrium speed of the ring
# ---------------------------------
# In uniform flow the acceleration vanishes when `1 - (v / v0)^delta = ((s0 + v T) / s)^2`, with the gap
# `s = length / n_cars - car_length`. Rings with more cars leave smaller gaps, so the final speed should fall with the
# number of cars and match the root of that equation.

v0, T, s0, delta, car_length, length = 30.0, 1.5, 2.0, 4.0, 5.0, 500.0


def equilibrium(gap):
    """Return the speed at which the desired gap equals the gap."""
    return brentq(lambda v: 1 - (v / v0) ** delta - ((s0 + v * T) / gap) ** 2, 0.0, v0)


counts = np.arange(10, 61, 5)
finals = [
    Simulation(create(MODEL, n_cars=int(n)), dt=1.0).run(until=200).trajectory_['cars.v'][-1].mean() for n in counts
]
grid = np.arange(10, 61)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(grid, [equilibrium(length / n - car_length) for n in grid], color='black', label='equilibrium of the IDM')
ax.plot(counts, finals, 'o', label='simulation, mean speed at 200 s')
ax.set_xlabel('number of cars on the ring')
ax.set_ylabel('speed (m/s)')
ax.set_title('Denser rings settle at lower speeds')
ax.legend()
plt.show()
