"""
# Order book with imitation

Four hundred traders on a grid quote limit orders for one share, as fundamentalists who expect the price to return to
its value of 100 or as chartists who expect the last move to go on. The order book clears their orders, and the price
swings around the fundamental value while traders copy the strategy of more successful neighbours, so the share of
chartists falls.
"""

# %%
# Every step every trader forecasts the price with its strategy and a 2% noise, and bids its forecast for one share
# when it is above the last price or asks it when it is below. A matching law pairs the orders with the limit order
# book of the economics realizations, with price-time priority, and every trade moves money and a share between the
# two traders. Then every trader may copy the strategy of a random neighbour whose wealth grew more. Half the traders
# start as chartists. The blueprint shows the price and the share of chartists.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'order_book'
STEPS = 150

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/reporters/book.price', name='price', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(origin='/chartists', name='share of chartists', axis_x=rrb.TimeAxis(view_range=whole_run)),
        row_shares=[2, 1],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, chartists=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=STEPS)

# %%
# Chartists make the market volatile
# ----------------------------------
# Fundamentalists pull the price back toward the fundamental value, while chartists extrapolate its moves (Caiani et
# al. 2016 ch. 3, Lux and Marchesi 1999). Without imitation, the standard deviation of the log returns should grow
# with the share of chartists, and the price should wander further from 100. Since fundamentalists buy low and sell
# high, they earn more, and imitation should drive the share of chartists down over the run.

shares = (0.0, 0.25, 0.5, 0.75)
volatility, deviation = [], []
for share in shares:
    run = Simulation(create(MODEL, chartists=share, imitation=0.0, random_state=0), random_state=0).run(until=STEPS)
    prices = run.trajectory_['book.price'][1:]
    volatility.append(np.diff(np.log(prices)).std())
    deviation.append(np.abs(np.log(prices / 100)).mean())
fig, (returns, chartists) = plt.subplots(1, 2, figsize=(11, 4))
returns.plot(shares, volatility, marker='o', label='standard deviation of log returns')
returns.plot(shares, deviation, marker='s', label='mean distance of log price from log 100')
returns.set_xlabel('share of chartists, without imitation')
returns.set_ylabel('size')
returns.set_title('Chartists amplify the swings of the price')
returns.legend()
trajectory = simulation.trajectory_
chartists.plot(trajectory.times, trajectory['chartists'])
chartists.set_xlabel('step')
chartists.set_ylabel('share of chartists')
chartists.set_title('Imitation favours fundamentalists')
fig.tight_layout()
plt.show()
