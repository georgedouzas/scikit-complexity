"""
# Success to the successful

Two competitors share a budget, and the more successful one gets more than its proportional share. Each share feeds
the stock that earns the next share, so two reinforcing loops push the competitors apart. The competitor that starts
larger ends with the whole budget, and the other one disappears.
"""

# %%
# The first competitor starts with 55 units of resources and the second with 45. The budget is 10 units a year and
# resources last 10 years, and the run lasts 100 years. The blueprint shows the two stocks.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import loops
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'success_to_the_successful'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/contest/a', name='first competitor a'),
        rrb.TimeSeriesView(origin='/contest/b', name='second competitor b'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1.0, observers=[observer]).run(100)

# %%
# The winner takes all
# --------------------
# The loop analysis finds two reinforcing loops, one through each competitor's allocation, and two balancing loops of
# depreciation. Starting runs from several splits of the same 100 units shows that the symmetric split is unstable,
# and whoever starts ahead wins.

polarities = [loop.polarity for loop in loops(create(MODEL)).loops]
summary = ', '.join(f'{polarities.count(kind)} {kind}' for kind in ('reinforcing', 'balancing'))

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (stocks, splits) = plt.subplots(1, 2, figsize=(10, 4))
stocks.plot(times, trajectory['contest.a'], label='first competitor')
stocks.plot(times, trajectory['contest.b'], label='second competitor')
stocks.set_xlabel('years')
stocks.set_ylabel('resources')
stocks.set_title('The resources of the competitors')
stocks.legend()
for a in (30.0, 45.0, 49.0, 51.0, 55.0, 70.0):
    run = Simulation(create(MODEL, a=a, b=100.0 - a), dt=1.0).run(100)
    splits.plot(times, run.trajectory_['contest.success'], label=f'a starts at {a:.0f}')
splits.set_xlabel('years')
splits.set_ylabel('share of the first competitor')
splits.set_title('The share of the first competitor')
splits.legend()
fig.suptitle(f'Feedback loops: {summary}')
fig.tight_layout()
plt.show()
