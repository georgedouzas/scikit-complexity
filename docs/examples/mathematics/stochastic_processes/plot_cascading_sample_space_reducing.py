"""
# Cascading sample-space-reducing process

A ball on a staircase of states jumps down to a random lower state and, wherever it lands, multiplies into `mu` balls,
each of which keeps falling. The cascade ends when every ball reaches the bottom, and the visits of the states follow
the power law `i^(-mu)`, whatever the multiplication factor, not just the Zipf exponent 1.
"""

# %%
# A hundred cascades with `mu = 1.5` fall from the top of 100 states. The system logs the number of balls, which grows
# as the cascades branch and falls to 0 as they reach the bottom, and the blueprint shows it.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.stochastic_processes import CascadingSampleSpaceReducingSystem

SYSTEM = 'cascading_sample_space_reducing'


def visits(trajectory, n_states):
    """Return the number of visits of every state from 1 to `n_states`, the levels of the balls of every frame."""
    levels = np.concatenate(
        [trajectory.agents('balls', frame).values['level'] for frame in range(len(trajectory.times))],
    )
    return np.bincount(levels.astype(int), minlength=n_states + 1)[1:]


whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/balls', name='number of balls', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(CascadingSampleSpaceReducingSystem(), dt=1, observers=[observer], random_state=0)

# %%
# Any exponent
# ------------
# The visits to state `i`, relative to those of state 1, should fall as `i^(-mu)` well below the top state: more
# flatly than Zipf's law when balls are lost on the way, `mu < 1`, as Zipf's law for `mu = 1`, and more steeply when
# the balls multiply.

fig, ax = plt.subplots(figsize=(6.5, 4.5))
states = np.arange(1, 51)
for mu, n_cascades in ((0.5, 3000), (1.0, 1000), (1.5, 200), (2.0, 50)):
    run = simulate(CascadingSampleSpaceReducingSystem(n_cascades=n_cascades, mu=mu), dt=1, random_state=1)
    counts = visits(run, 100)[:50]
    (line,) = ax.loglog(states, counts / counts[0], '.', label=f'mu = {mu}')
    ax.loglog(states, states ** (-mu), color=line.get_color(), linewidth=0.8)
ax.set_xlabel('state i')
ax.set_ylabel('visits relative to state 1')
ax.set_title('Visits fall as i^-mu (lines)')
ax.legend()
plt.show()
