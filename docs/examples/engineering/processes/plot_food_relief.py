"""
# Food relief system

A disaster leaves 100000 people in need of food. Relief food is shipped to a warehouse through a pipeline with a
two-week third-order delay, so almost nothing reaches the people at first, and the warehouse stock overshoots its
target before the distribution settles at the need.
"""

# %%
# The disaster strikes on day 10, each person needs half a kilogram a day, and the agency targets a warehouse cover of
# seven days of need. The run lasts 120 days. The blueprint shows the food in transit and in the warehouse.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'food_relief'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/relief/in_transit', name='food in transit', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/relief/warehouse', name='warehouse', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(create(MODEL), dt=0.5, observers=[observer]).run(120)

# %%
# The gap during the shipping delay
# ---------------------------------
# The shipments jump at the onset to fill the pipeline and the warehouse, but the arrivals follow the Erlang response
# of the third-order delay. The distribution meets the need only once enough food has arrived, and the unmet need in
# between is the cost of the delay.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
need, distribution = trajectory['relief.need'], trajectory['relief.distribution']
fig, (flows, gap) = plt.subplots(1, 2, figsize=(10, 4))
flows.plot(times, trajectory['relief.shipping'], label='shipping')
flows.plot(times, trajectory['relief.arrivals'], label='arrivals')
flows.plot(times, distribution, label='distribution')
flows.set_xlabel('days')
flows.set_ylabel('tonnes per day')
flows.set_title('Relief flows')
flows.legend()
gap.fill_between(times, distribution, need, color='tab:red', alpha=0.3, label='unmet need')
gap.plot(times, need, color='black', label='need')
gap.set_xlabel('days')
gap.set_ylabel('tonnes per day')
gap.set_title('Need and distribution')
gap.legend()
fig.tight_layout()
plt.show()
