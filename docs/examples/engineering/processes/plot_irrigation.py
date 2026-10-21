"""
# Irrigation system

A reservoir fed by a seasonal river irrigates a crop through leaky canals, and an aquifer below backs it up. In the
dry season the reservoir runs low, the farmers pump groundwater, and the aquifer is drawn down from year to year.
"""

# %%
# The river brings 0.5 million cubic metres a day on average, with a yearly cycle of relative amplitude 0.9, and the
# crops need 0.4 a day. The run lasts two years. The blueprint shows the reservoir, the root zone, and the aquifer.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'irrigation'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/scheme/reservoir', name='reservoir', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/scheme/soil', name='root zone', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/scheme/groundwater', name='aquifer', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=5)
simulation = Simulation(create(MODEL), dt=1.0, observers=[observer]).run(730)

# %%
# Groundwater as a buffer
# -----------------------
# The release from the reservoir covers the demand while the reservoir holds water. When it runs low the release
# falls, the pumping takes over up to the pump capacity, and the crops come under water stress only when both fall
# short.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (stocks, flows) = plt.subplots(1, 2, figsize=(10, 4))
stocks.plot(times, trajectory['scheme.reservoir'], label='reservoir')
stocks.plot(times, trajectory['scheme.groundwater'], label='aquifer')
stocks.set_xlabel('days')
stocks.set_ylabel('million cubic metres')
stocks.set_title('Water stocks')
stocks.legend()
flows.plot(times, trajectory['scheme.irrigation'], label='irrigation')
flows.plot(times, trajectory['scheme.groundwater_pumping'], label='pumping')
flows.plot(times, trajectory['scheme.crop_water'], '--', label='crop water use')
flows.set_xlabel('days')
flows.set_ylabel('million cubic metres per day')
flows.set_title('Water flows')
flows.legend()
fig.tight_layout()
plt.show()
