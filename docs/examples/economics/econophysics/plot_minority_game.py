"""
# Minority game

An odd number of agents repeatedly choose one of two sides, and those on the minority side win. Each agent holds two
fixed strategies, which map the last `m` winning sides to an action, and plays the one that would have won most
often. The attendance fluctuates around zero, and its volatility depends only on `alpha = 2^m / N`: herding makes it
larger than for random play when the agents remember little, and coordination makes it much smaller near
`alpha_c ~ 0.34`.
"""

# %%
# Fifty-one agents with memory `m = 4`, so `alpha ~ 0.31`, play 300 rounds. The blueprint shows the attendance
# `A = sum_i a_i`, the number of agents on the `+1` side minus those on the `-1` side.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'minority_game'
N, ROUNDS, TRANSIENT = 51, 300, 100

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters/game.attendance',
        name='attendance A',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
system = create(MODEL, n_agents=N, memory=4, random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=ROUNDS)

# %%
# Volatility against alpha
# ------------------------
# Random play gives `sigma^2 / N = 1`. The volatility of the attendance after a transient, for memories from 1 to 7,
# should lie above one for small `alpha`, dip to a minimum near `alpha_c ~ 0.34`, and approach one again for large
# `alpha`, the curve of Savit, Manuca, and Riolo. The evolutionary version, where every 50 rounds the poorest tenth of
# the agents take new strategies, keeps the volatility low.


def volatility(memory, **options):
    """Return `sigma^2 / N` of the attendance after the transient."""
    run = Simulation(create(MODEL, n_agents=N, memory=memory, random_state=0, **options), random_state=0)
    attendance = run.run(until=ROUNDS).trajectory_['game.attendance']
    return attendance[TRANSIENT:].var() / N


memories = np.arange(1, 8)
alphas = 2.0**memories / N
fig, (series, curve) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, simulation.trajectory_['game.attendance'], lw=0.8)
series.set_xlabel('round')
series.set_ylabel('attendance A')
series.set_title('Attendance for m = 4')
curve.loglog(alphas, [volatility(memory) for memory in memories], 'o-', label='minority game')
curve.loglog(alphas, [volatility(memory, evolutionary=True) for memory in memories], 's--', label='evolutionary')
curve.axhline(1, color='grey', linestyle=':', label='random play')
curve.axvline(0.34, color='black', linestyle='--', lw=0.8, label='alpha_c ~ 0.34')
curve.set_xlabel('alpha = 2^m / N')
curve.set_ylabel('sigma^2 / N')
curve.set_title('Volatility of the attendance')
curve.legend()
fig.tight_layout()
plt.show()
