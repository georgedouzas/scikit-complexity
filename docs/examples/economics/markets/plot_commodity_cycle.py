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
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation, create_system

SYSTEM = 'commodity_cycle'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/market/inventory', name='inventory'),
        rrb.TimeSeriesView(origin='/market/capacity', name='capacity'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
simulation = create_simulation(SYSTEM).set_params(dt=0.25, observers=[observer]).prepare().run(until=60)

# %%
# The price leads the capacity
# ----------------------------
# The loop analysis finds the balancing loops of the market. The longest one runs from the capacity through the
# production to the inventory, whose price sets the capacity orders, and back through the construction delay and the
# completions to the capacity, and its delay drives the cycle. The price peaks first, the capacity orders follow, and
# the capacity peaks years later.

polarities = loops(create_system(SYSTEM)).polarities
summary = ', '.join(f'{polarities.count(kind)} {kind}' for kind in ('reinforcing', 'balancing'))

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (price, capacity) = plt.subplots(1, 2, figsize=(10, 4))
price.plot(times, trajectory.values('market.price'), label='price')
price.plot(times, trajectory.values('market.expected_price'), '--', label='expected price')
price.set_xlabel('years')
price.set_ylabel('price relative to the reference')
price.set_title('The price')
price.legend()
capacity.plot(times, trajectory.values('market.capacity_orders'), label='capacity orders')
capacity.plot(times, trajectory.values('market.completions'), '--', label='completions')
capacity.plot(times, trajectory.values('market.retirements'), ':', label='retirements')
capacity.set_xlabel('years')
capacity.set_ylabel('capacity per year')
capacity.set_title('The flows of capacity')
capacity.legend()
fig.suptitle(f'Feedback loops: {summary}')
fig.tight_layout()
plt.show()
