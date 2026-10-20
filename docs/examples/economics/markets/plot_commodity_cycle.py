"""
# Commodity cycle with capacity delay

A commodity market sets its price from the inventory coverage. A higher price draws more production from the existing
capacity and leads producers to order new capacity, which arrives only after a construction delay. By the time it
arrives the price has already fallen, so capacity overshoots and the price and the capacity cycle.
"""

# %%
# The market starts in equilibrium, with a capacity of 100 units a year and a quarter of a year of inventory. Demand
# rises by 10% in year 2, and the run lasts 60 years. The blueprint shows the inventory above the capacity.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import loops
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'commodity_cycle'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/market/inventory', name='inventory'),
        rrb.TimeSeriesView(origin='/market/capacity', name='capacity'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=0.25, observers=[observer]).run(60)

# %%
# The price leads the capacity
# ----------------------------
# The loop analysis finds the balancing loops of the market. The longest one runs from the price through the expected
# price, the capacity orders, the construction delay, and the capacity back to the inventory and the price, and its
# delay drives the cycle. The price peaks first, the capacity orders follow, and the capacity peaks years later.

polarities = [loop.polarity for loop in loops(create(MODEL)).loops]
summary = ', '.join(f'{polarities.count(kind)} {kind}' for kind in ('reinforcing', 'balancing'))

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (price, capacity) = plt.subplots(1, 2, figsize=(10, 4))
price.plot(times, trajectory['market.price'], label='price')
price.plot(times, trajectory['market.expected_price'], '--', label='expected price')
price.set_xlabel('years')
price.set_ylabel('price relative to the reference')
price.set_title('The price')
price.legend()
capacity.plot(times, trajectory['market.capacity_orders'], label='capacity orders')
capacity.plot(times, trajectory['market.completions'], '--', label='completions')
capacity.plot(times, trajectory['market.retirements'], ':', label='retirements')
capacity.set_xlabel('years')
capacity.set_ylabel('capacity per year')
capacity.set_title('The flows of capacity')
capacity.legend()
fig.suptitle(f'Feedback loops: {summary}')
fig.tight_layout()
plt.show()
