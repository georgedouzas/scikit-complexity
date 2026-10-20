"""
# Kolkata Paise Restaurant

Every evening as many diners as restaurants each pick a restaurant, and each restaurant serves one of the diners who
come. Choosing at random leaves about a third of the restaurants empty, while diners who stay where they were served
and leave crowded restaurants with the stochastic crowd avoiding strategy fill about four fifths of them.
"""

# %%
# Two hundred crowd avoiding diners return to last evening's restaurant with probability one over the crowd they
# found there, and otherwise move to another random restaurant. The blueprint shows the utilization, the fraction of
# restaurants that serve a meal.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'kolkata_restaurant'
EVENINGS = 100

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters/restaurants.utilization',
        name='utilization',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, n_agents=200, strategy='crowd_avoiding', random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=EVENINGS)

# %%
# Utilization of the two strategies
# ---------------------------------
# A restaurant chosen by none of `N` random diners has probability `(1 - 1/N)^N ~ 1/e`, so random choice uses
# `1 - 1/e ~ 0.632` of the restaurants. Ghosh, Chatterjee, Mitra, and Chakrabarti found about 0.8 for the stochastic
# crowd avoiding strategy, reached within a few evenings.

random = Simulation(create(MODEL, n_agents=200, strategy='random', random_state=0), random_state=0).run(EVENINGS)
fig, ax = plt.subplots(figsize=(7, 4))
times = simulation.trajectory_.times
for label, run in [('random', random), ('crowd avoiding', simulation)]:
    values = run.trajectory_['restaurants.utilization']
    ax.plot(times, values, lw=0.8, label=f'{label}, mean {np.nanmean(values[20:]):.3f}')
ax.axhline(1 - 1 / np.e, color='C0', linestyle='--', label='1 - 1/e')
ax.axhline(0.8, color='C1', linestyle='--', label='0.8')
ax.set_ylim(0.5, 1)
ax.set_xlabel('evening')
ax.set_ylabel('utilization')
ax.set_title('Fraction of restaurants that serve a meal')
ax.legend()
fig.tight_layout()
plt.show()
