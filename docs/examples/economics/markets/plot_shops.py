"""
# Shops and digital markets

Sixteen shops on a grid sell to a shopper on every cell. When everyone shops online, distance no longer shelters a
shop, the cheapest one takes every shopper, and within a few steps all but two of the sixteen shops close, while
shops that compete in person for nearby shoppers mostly survive.
"""

# %%
# Every step every shopper wants one unit and buys it from the shop of the lowest price plus travel cost, where the
# travel cost is one per cell in person and zero online. Shops search for the price of the largest profit by trial and
# error, pay a fixed cost of 20 every step, and close when their money runs out. In this run everyone shops online.
# The blueprint shows the number of open shops and, beside it, the shops on the grid coloured by price.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'shops'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
open_shops = rrb.TimeSeriesView(origin='/open_shops', name='open shops', axis_x=rrb.TimeAxis(view_range=whole_run))
town = rrb.Spatial2DView(
    origin='/',
    contents=['/shoppers/**', '/shops/**'],
    name='town',
    overrides={
        'shoppers': rr.Points2D.from_fields(colors=[90, 90, 90], radii=0.2).visualizer(),
        'shops': rr.Points2D.from_fields(radii=0.6).visualizer(),
    },
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(open_shops, town, column_shares=[3, 2]),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, online=1.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=100)

# %%
# Distance protects shops
# -----------------------
# Hamill and Gilbert (2016 sec. 6.3-6.4) contrast local shops, each with a market of the shoppers nearby, with a digital
# market where location does not matter. With more shoppers online fewer shops should survive and the largest shop
# should take a larger share of the sales. The price should fall in the digital market, where shops undercut each
# other, and stay high where travel costs give every shop a small monopoly.

shares = np.linspace(0.0, 1.0, 5)
survivors, leaders, prices = [], [], []
for share in shares:
    system = create(MODEL, online=share, random_state=0)
    run = Simulation(system, random_state=0, observables=describe(MODEL).observables_of(system)).run(until=100)
    survivors.append(run.trajectory_['open_shops'][-1])
    leaders.append(run.trajectory_['concentration'][-1])
    prices.append(run.trajectory_['mean_price'][50:].mean())
fig, (count, leader, price) = plt.subplots(1, 3, figsize=(13, 4))
count.plot(shares, survivors, marker='o')
count.set_xlabel('share of shoppers online')
count.set_ylabel('open shops after 100 steps')
count.set_title('Online shopping closes shops')
leader.plot(shares, leaders, marker='o', color='C1')
leader.set_xlabel('share of shoppers online')
leader.set_ylabel('share of sales of the largest shop')
leader.set_title('One shop takes the market')
price.plot(shares, prices, marker='o', color='C2')
price.set_xlabel('share of shoppers online')
price.set_ylabel('mean price over the last 50 steps')
price.set_title('Competition online lowers prices')
fig.tight_layout()
plt.show()
