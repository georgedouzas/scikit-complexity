"""
# Cocoa boom and bust

Cocoa plantations expand while they pay, and farmers thin the shading trees to raise the yield. The thinning lets the
insects multiply, the yield collapses, and the planted area busts within two decades. A subsidy that keeps the shade
and controls the insects avoids the bust.
"""

# %%
# The run starts from ten thousand hectares of cocoa in a forest of half a million and lasts fifty years without a
# subsidy. The blueprint shows the planted area and the insect attack index.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cocoa'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/cocoa/area', name='planted area', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/cocoa/insects', name='insect attack', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=0.25, observers=[observer]).run(50)

# %%
# Subsidy coverage decides the bust
# ---------------------------------
# Without a subsidy the area peaks after about twelve years and falls to almost nothing as the yield collapses. A
# subsidy slows the thinning and pays for insect control, so the area should keep growing for longer, and with full
# coverage it should level off near the limit of the forest instead of busting.

fig, (area, crop) = plt.subplots(1, 2, figsize=(10, 4))
runs = {0.0: simulation.trajectory_}
runs.update(
    {subsidy: Simulation(create(MODEL, subsidy=subsidy), dt=0.25).run(50).trajectory_ for subsidy in (0.6, 1.0)},
)
for subsidy, run in runs.items():
    times = np.asarray(run.times)
    area.plot(times, run['cocoa.area'], label=f'subsidy {subsidy:.0%}')
    crop.plot(times, run['cocoa.cocoa_yield'], label=f'subsidy {subsidy:.0%}')
area.set_xlabel('years')
area.set_ylabel('planted area (thousand ha)')
area.set_title('Planted area')
area.legend()
crop.set_xlabel('years')
crop.set_ylabel('yield (100 kg per ha)')
crop.set_title('Cocoa yield')
crop.legend()
fig.tight_layout()
plt.show()
