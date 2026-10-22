"""
# Ridesharing

Thirty cabs serve riders who appear at random on a 30 by 30 city grid. When riders request trips a little faster than
the fleet can serve them, every cab stays busy, the queue of waiting riders keeps growing, and the mean wait climbs
step after step.
"""

# %%
# Every step new riders appear, 1.2 on average, at random cells with random destinations. A matching law pairs the
# waiting riders, the longest waiting first, with the nearest idle cabs, and every cab drives one block a step to its
# rider, collects the fare as a transfer, and drives the rider to its destination. The blueprint shows the number of
# waiting riders and the city, with the riders coloured by how long they have waited and the cabs by their task.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'ridesharing'
STEPS = 200

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
queue = rrb.TimeSeriesView(origin='/queue', name='waiting riders', axis_x=rrb.TimeAxis(view_range=whole_run))
city = rrb.Spatial2DView(
    origin='/',
    contents=['/riders/**', '/cabs/**'],
    name='city',
    overrides={
        'riders': rr.Points2D.from_fields(radii=0.35).visualizer(),
        'cabs': rr.Points2D.from_fields(radii=0.6).visualizer(),
    },
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(queue, city, column_shares=[3, 2]),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, demand=1.2, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=STEPS)

# %%
# Waiting near saturation
# -----------------------
# A trip takes the cab to the rider and on to a random destination, about 20 blocks on a 30 by 30 grid, plus the
# drive to the rider, so 30 cabs serve about one rider a step. Below that demand the cabs are often idle and riders
# wait about as long as the nearest cab takes to reach them. Near and above it, as in a queue near saturation, the
# utilisation of the fleet should approach one and the mean wait should rise steeply (Carmichael and Hadzikadic 2019
# ch. 7).

demands = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
waits, utilisations = [], []
for demand in demands:
    system = create(MODEL, demand=demand, random_state=0)
    run = Simulation(system, random_state=0, observables=describe(MODEL).observables_of(system)).run(until=STEPS)
    waits.append(np.nanmean(run.trajectory_['mean_wait'][STEPS // 2 :]))
    utilisations.append(run.trajectory_['utilisation'][STEPS // 2 :].mean())
fig, (wait, busy) = plt.subplots(1, 2, figsize=(11, 4))
wait.plot(demands, waits, marker='o')
wait.set_xlabel('new riders per step')
wait.set_ylabel('mean wait of the waiting riders, in steps')
wait.set_title('Waits rise steeply near saturation')
busy.plot(demands, utilisations, marker='o', color='C1')
busy.axhline(1.0, color='grey', linestyle=':')
busy.set_xlabel('new riders per step')
busy.set_ylabel('share of busy cabs')
busy.set_title('The fleet saturates')
fig.tight_layout()
plt.show()
