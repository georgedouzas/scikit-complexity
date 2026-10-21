"""
# Sample-space-reducing process

A ball on a staircase of ten states can only jump down, to a random lower state, and restarts at the top once it
reaches the bottom: its path is a series of descending staircases. Because high states are skipped more often than
low ones, the visits follow Zipf's law, the frequency of state `i` falling as `1 / i`.
"""

# %%
# Twenty balls jump for 40 steps with `lam = 1`, so every jump goes down. The model logs only the number of balls, so
# after the run the level of one ball is logged as a time series, and the blueprint puts it first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sample_space_reducing'


def levels(trajectory):
    """Return the levels of the balls, one row per step."""
    return np.array([trajectory.agents('balls', frame).values['level'] for frame in range(len(trajectory.times))])


whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/levels', name='level of one ball', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1, observers=[observer], random_state=0).run(until=40)
path = levels(simulation.trajectory_)
observer.recording_.send_columns(
    'levels/ball_0',
    indexes=[rr.TimeColumn('sim_time', duration=simulation.trajectory_.times)],
    columns=rr.Scalars.columns(scalars=path[:, 0]),
)
observer.recording_.flush()

# %%
# Zipf's law and its noisy versions
# ---------------------------------
# Over 200 balls and 1000 steps on 100 states, the share of visits to state `i`, below the restart state 100, should
# fall as `i^(-lam)`: as Zipf's `1 / i` for strictly reducing jumps, `lam = 1`, more flatly when a share `1 - lam` of
# the jumps may go anywhere, and uniformly for `lam = 0`.

fig, ax = plt.subplots(figsize=(6.5, 4.5))
states = np.arange(1, 100)
for lam in (1.0, 0.7, 0.4, 0.0):
    run = Simulation(create(MODEL, n_balls=200, n_states=100, lam=lam), dt=1, random_state=1).run(until=1000)
    visits = np.bincount(levels(run.trajectory_)[100:].astype(int).ravel(), minlength=101)[1:100]
    (line,) = ax.loglog(states, visits / visits.sum(), '.', label=f'lam = {lam}')
    reference = states ** (-lam)
    ax.loglog(states, reference / reference.sum(), color=line.get_color(), linewidth=0.8)
ax.set_xlabel('state i')
ax.set_ylabel('share of visits')
ax.set_title('Visits fall as i^-lam (lines)')
ax.legend()
plt.show()
