"""
# Cascade classifier

Particles fed into the middle of a column of ten stages are carried up or down by the air flow from stage to stage.
With 60% of the material going up at every stage, the stages fill to a steady profile, and most of the feed leaves
at the top as the fine product.
"""

# %%
# The model is created by name from the catalog with ten stages, the upward fraction `k = 0.6`, and a unit feed into
# the middle stage, and runs for 50 time units. The blueprint shows the accumulated fine and coarse products in one time
# series view, where the fine product grows faster.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cascade_classifier'

products = rrb.TimeSeriesView(
    origin='/',
    contents=['/fine/m', '/coarse/m'],
    name='products',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    products,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=50)

# %%
# Barsky's fractional extraction
# ------------------------------
# At steady state the share of the feed that leaves as the fine product should be
# `F_f = (1 - ((1 - k) / k)^(z + 1 - i)) / (1 - ((1 - k) / k)^(z + 1))` for `z` stages fed at stage `i` from the top.
# The rate of growth of the fine product at the end of a long run, against the unit feed, should follow this curve for
# every upward fraction `k`.

z = 10
fractions = np.linspace(0.3, 0.7, 9)
shares = []
for k in fractions:
    run = Simulation(create(MODEL, k=float(k)), dt=1.0).run(until=300).trajectory_
    shares.append(run['fine.m'][-1] - run['fine.m'][-2])
grid = np.linspace(0.3, 0.7, 200)
feed = z // 2
ratio = (1 - grid) / grid
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(grid, (1 - ratio ** (z + 1 - feed)) / (1 - ratio ** (z + 1)), color='black', label='Barsky')
ax.plot(fractions, shares, 'o', label='simulation')
ax.set_xlabel('upward fraction k')
ax.set_ylabel('share of the feed in the fine product')
ax.set_title('Fractional extraction of a ten-stage cascade')
ax.legend()
plt.show()
