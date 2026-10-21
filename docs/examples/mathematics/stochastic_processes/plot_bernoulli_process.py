"""
# Bernoulli process

Each of six sequences adds a success with probability 1/2 at every step, so the success counts climb as noisy
staircases around the line `p t`, spreading apart as the square root of the number of trials.
"""

# %%
# Six sequences run for 100 trials with `p = 0.5`. The model logs only the number of sequences, so after the run the
# success count of every sequence is logged as a time series, and the blueprint puts it first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy import stats

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bernoulli_process'


def successes(trajectory):
    """Return the success counts, one row per step and one column per sequence."""
    return np.array([trajectory.agents('trials', frame).values['successes'] for frame in range(len(trajectory.times))])


whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/successes', name='successes', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, n_sequences=6), dt=1, observers=[observer], random_state=0).run(until=100)
counts = successes(simulation.trajectory_)
for sequence in range(6):
    observer.recording_.send_columns(
        f'successes/sequence_{sequence}',
        indexes=[rr.TimeColumn('sim_time', duration=simulation.trajectory_.times)],
        columns=rr.Scalars.columns(scalars=counts[:, sequence]),
    )
observer.recording_.flush()

# %%
# Binomial counts
# ---------------
# Over 2000 sequences the number of successes after `n` trials should follow the binomial distribution `B(n, p)`,
# here with mean 50 and variance 25 at `n = 100`, for `p = 0.5` and for `p = 0.2`.

fig, ax = plt.subplots(figsize=(6.5, 4.5))
grid = np.arange(0, 80)
for p, color in ((0.2, 'C1'), (0.5, 'C0')):
    final = successes(Simulation(create(MODEL, n_sequences=2000, p=p), dt=1, random_state=1).run(until=100).trajectory_)
    ax.hist(final[-1], bins=np.arange(-0.5, 80.5), density=True, alpha=0.5, color=color, label=f'simulated, p = {p}')
    ax.plot(grid, stats.binom.pmf(grid, 100, p), '.-', color=color, markersize=3, label=f'B(100, {p})')
ax.set_xlabel('successes after 100 trials')
ax.set_ylabel('probability')
ax.set_title('Success counts are binomial')
ax.legend()
plt.show()
