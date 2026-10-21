"""
# Delayed SIRS in two cities

A disease with a fixed infectious period and a fixed period of immunity breaks out again and again: each outbreak burns
through the susceptibles, the immune protect the city for a while, and when their immunity ends the next outbreak
comes. A few visitors between two cities lock their outbreaks, here out of phase: the cities take turns.
"""

# %%
# The model is created by name from the catalog with the infection starting in the first city and six steps of
# immunity. The blueprint shows the infected fraction of each city in one time series view over the last 60 steps.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'delayed_sirs'

last_steps = rrb.TimeRange(
    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-60),
    end=rrb.TimeRangeBoundary.cursor_relative(),
)
cities = rrb.TimeSeriesView(
    origin='/',
    contents=['/first/x', '/second/x'],
    name='infected fraction',
    defaults=[rr.SeriesLines.from_fields(widths=2.5)],
    axis_x=rrb.TimeAxis(view_range=last_steps),
)
blueprint = rrb.Blueprint(
    cities,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, first=0.01, second=0.0, recovery=6, e=0.005)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=200)

# %%
# Recurrent and locked outbreaks
# ------------------------------
# The outbreaks should recur with a fixed period longer than `recovery + 1` steps, the time to clear the infection and
# lose the immunity. The second city catches the disease from the visitors and its outbreaks should lock to those of
# the first half a period apart, so the two series are anticorrelated at zero lag and correlated at half a period.

trajectory = simulation.trajectory_
first, second = trajectory['first.x'], trajectory['second.x']
late = trajectory.times > 100
lags = np.arange(-12, 13)
correlation = [np.corrcoef(first[late], np.roll(second, lag)[late])[0, 1] for lag in lags]
fig, (series, cross) = plt.subplots(1, 2, figsize=(11, 4))
series.plot(trajectory.times, first, label='first city')
series.plot(trajectory.times, second, label='second city')
series.set_xlim(100, 200)
series.set_xlabel('step')
series.set_ylabel('infected fraction')
series.set_title('Recurrent outbreaks, the cities take turns')
series.legend()
cross.bar(lags, correlation)
cross.set_xlabel('lag of the second city (steps)')
cross.set_ylabel('correlation')
cross.set_title(
    f'Correlation {correlation[12]:.2f} at lag 0, {max(correlation):.2f} at lag {lags[np.argmax(correlation)]}',
)
fig.tight_layout()
plt.show()
