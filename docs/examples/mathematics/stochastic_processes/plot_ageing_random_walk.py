"""
# Ageing random walk

Each walker steps up with a probability that falls with the number of up steps it has already taken, so the walk
remembers its history: after a few early up steps the walkers drift downward almost like straight lines, their rare
up steps growing only as a power of time below 1.
"""

# %%
# Six walkers start at 0 with `p = 0.5` and `nu = 1` and take 299 steps. The model logs only the number of walkers, so
# after the run the positions of the six walkers are logged as time series, and the blueprint puts them first, as the
# primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'ageing_random_walk'


def states(trajectory, name):
    """Return a state of the walkers, one row per step."""
    return np.array([trajectory.agents('walkers', frame).values[name] for frame in range(len(trajectory.times))])


whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/positions', name='positions', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, n_walkers=6), dt=1, observers=[observer], random_state=0).run(until=299)
positions = states(simulation.trajectory_, 'x')
for walker in range(6):
    observer.recording_.send_columns(
        f'positions/walker_{walker}',
        indexes=[rr.TimeColumn('sim_time', duration=simulation.trajectory_.times)],
        columns=rr.Scalars.columns(scalars=positions[:, walker]),
    )
observer.recording_.flush()

# %%
# Sublinear growth of the up steps
# --------------------------------
# After `k` up steps the next step goes up with probability `p / (1 + k)^nu`, so the mean number of up steps after `n`
# steps should grow as `n^(1 / (1 + nu))`: as `√n` for `nu = 1`, as `n^(2/3)` for `nu = 0.5`, and as `n^(1/3)` for
# `nu = 2`, approached as straight lines of those slopes on log-log axes at long times, over 500 walkers and 1000 steps.

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for nu in (0.5, 1.0, 2.0):
    run = Simulation(create(MODEL, n_walkers=500, nu=nu), dt=1, random_state=1).run(until=1000).trajectory_
    ups = states(run, 'ups').mean(axis=1)
    (line,) = ax.loglog(run.times[1:], ups[1:], label=f'nu = {nu}')
    ax.loglog(
        run.times[10:],
        ups[-1] * (run.times[10:] / run.times[-1]) ** (1 / (1 + nu)),
        '--',
        color=line.get_color(),
    )
ax.set_xlabel('step n')
ax.set_ylabel('mean number of up steps')
ax.set_title('Up steps grow as n^(1 / (1 + nu)) (dashed)')
ax.legend()
plt.show()
