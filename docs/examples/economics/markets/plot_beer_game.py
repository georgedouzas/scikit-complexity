"""
# Beer distribution game

A retailer, a wholesaler, a distributor, and a factory pass orders upstream and beer downstream, each through a delay.
Every echelon orders by Sterman's anchoring and adjustment rule, which underweights the beer already on order. When
customer demand steps from 4 to 8 cases a week, the orders swing wider at every echelon upstream, the bullwhip effect.
"""

# %%
# Every echelon starts with 12 cases and the chain ships 4 cases a week. Demand doubles in week 5, and the run lasts
# 36 weeks. The blueprint shows the inventory of every echelon.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.markets import BeerGameSystem

SYSTEM = 'beer_game'
ECHELONS = ('retailer', 'wholesaler', 'distributor', 'factory')

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/',
        contents=[f'/{echelon}/inventory' for echelon in ECHELONS],
        name='inventories',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(BeerGameSystem(), observers=[observer])

# %%
# The bullwhip effect
# -------------------
# The orders of each echelon peak later and higher than the orders it receives. The peak orders grow from the
# retailer to the factory, although customer demand only doubles.

times = np.asarray(trajectory.times)
fig, (orders, peaks) = plt.subplots(1, 2, figsize=(10, 4))
orders.plot(times, trajectory.get_values('customers.orders'), color='black', linestyle=':', label='customer demand')
for echelon in ECHELONS:
    orders.plot(times, trajectory.get_values(f'{echelon}.orders'), label=echelon)
orders.set_xlabel('weeks')
orders.set_ylabel('cases per week')
orders.set_title('The orders of every echelon')
orders.legend()
peaks.bar(ECHELONS, [trajectory.get_values(f'{echelon}.orders').max() for echelon in ECHELONS])
peaks.axhline(8.0, color='black', linestyle=':', label='customer demand after the step')
peaks.set_ylabel('peak orders, cases per week')
peaks.set_title('The amplification of orders upstream')
peaks.legend()
fig.tight_layout()
plt.show()
