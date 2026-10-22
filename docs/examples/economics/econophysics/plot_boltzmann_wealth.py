"""
# Boltzmann wealth

Every agent starts with one unit of wealth and, each step, gives one unit to a random agent if it has any. Although
the rule treats everyone alike, chance alone concentrates the wealth: the distribution relaxes to an exponential,
Boltzmann-like law, with many agents at zero and a few rich ones, and the Gini coefficient climbs from zero to about
0.6.
"""

# %%
# Three hundred agents trade for 200 steps. The model has no inequality measure of its own, so an observable adds the
# Gini coefficient of the wealth of the traders, which the blueprint shows.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import gini
from skcomplex.core.model import Observable
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'boltzmann_wealth'
STEPS = 200

system = create(MODEL, n_agents=300)
observables = [Observable('gini', group='traders', variable='wealth', reduce=gini)]
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/gini', name='Gini coefficient', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, observers=[observer], random_state=0, observables=observables).run(until=STEPS)

# %%
# Stationary distribution of wealth
# ---------------------------------
# Pooling the wealth of all agents over the second half of the run, the probability of holding `w` units should fall
# by a constant factor per unit, a straight line on a logarithmic axis, as for a Boltzmann distribution of energy. The
# Lorenz curve, the share of wealth held by the poorest fraction of agents, should bow far below the line of equality.

trajectory = simulation.trajectory_
wealth = np.concatenate(
    [trajectory.agents('traders', frame).values['wealth'] for frame in range(STEPS // 2, STEPS + 1)],
)
counts = np.bincount(wealth.astype(int)) / len(wealth)
units = np.arange(len(counts))
slope, intercept = np.polyfit(units[1:8], np.log(counts[1:8]), 1)
final = np.sort(trajectory.agents('traders', STEPS).values['wealth'])
fig, (law, lorenz) = plt.subplots(1, 2, figsize=(10, 4))
law.semilogy(units, counts, 'o', label='simulation')
law.semilogy(units, np.exp(intercept + slope * units), 'k--', lw=0.8, label=f'exp(-w / {-1 / slope:.2f})')
law.set_xlabel('wealth w')
law.set_ylabel('probability P(w)')
law.set_title('An exponential distribution of wealth')
law.legend()
lorenz.plot(np.linspace(0, 1, len(final) + 1), np.r_[0, np.cumsum(final)] / final.sum(), label='Lorenz curve')
lorenz.plot([0, 1], [0, 1], 'k:', label='equality')
lorenz.set_xlabel('share of agents, poorest first')
lorenz.set_ylabel('share of wealth')
lorenz.set_title(f'Final Gini coefficient {gini(final):.2f}')
lorenz.legend()
fig.tight_layout()
plt.show()
