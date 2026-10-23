"""
# Wilhite trade networks

Traders with random bundles of two goods barter with their neighbours on a network, each step with the neighbour whose
marginal rate of substitution differs the most, while both gain. On a small world of ten groups of traders, chained by
shared end traders and a few long links, trade drives every trader towards one price. The view shows the spread of the
logarithms of the traders' rates falling towards zero.
"""

# %%
# The model is created by name with 200 traders in ten groups of 20 on a small world with five long links, and bundles
# of up to 100 units of each good. The blueprint shows the observable `dispersion`, the standard deviation of the
# logarithms of the rates, above the units traded in each step, over the whole run of 60 steps.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'wilhite_trade_networks'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/dispersion', name='price dispersion', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(
            origin='/reporters',
            contents=['/reporters/barter.trades'],
            name='units traded',
            axis_x=rrb.TimeAxis(view_range=whole_run),
        ),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, n_agents=200, topology='small_world', n_groups=10, shortcuts=5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=60)

# %%
# Prices against searches
# -----------------------
# Wilhite (2001) compared the four networks by how close they bring the traders' rates and how many searches they
# need. Isolated groups settle on different prices, so their dispersion stalls. On the complete network every trader
# searches all the others each step, and the traders all chase the few most extreme partners, so the price spreads
# slowly, as Gooding (2019) notes of perfect information. The linked groups and the small world should reach the lowest
# dispersion with a tenth of the searches of the complete network.

fig, ax = plt.subplots(figsize=(7, 4.5))
for topology in ('isolated', 'linked', 'small_world', 'complete'):
    network = create(MODEL, n_agents=200, topology=topology, n_groups=10, shortcuts=5, random_state=0)
    run = Simulation(network, random_state=0, observables=describe(MODEL).observables_of(network)).run(until=40)
    trajectory = run.trajectory_
    searches = np.concatenate([[0.0], np.cumsum(trajectory['barter.searches'][1:])])
    ax.loglog(searches[1:], trajectory['dispersion'][1:], label=topology)
ax.set_xlabel('cumulative searches')
ax.set_ylabel('dispersion of log rates of substitution')
ax.set_title('Small worlds unify prices with few searches')
ax.legend()
fig.tight_layout()
plt.show()
