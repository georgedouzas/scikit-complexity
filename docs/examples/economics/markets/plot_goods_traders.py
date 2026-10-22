"""
# Goods traders

Two hundred traders wander a grid and barter sugar, spice, and salt with their neighbours, each trade at the geometric
mean of the two traders' marginal rates of substitution. No auctioneer sets a price, yet the prices of different
trades converge, and the spread of the price of spice in sugar falls by more than a factor of ten.
"""

# %%
# Every trader holds random amounts of the three goods and values them with random Cobb-Douglas weights. Every step the
# traders wander to a random neighbouring cell, a matching law pairs every trader with a neighbour, and every pair
# trades small quantities while both gain. The blueprint shows the standard deviation of the logarithm of the price of
# spice in sugar over the trades of every step, and the traders on the grid, coloured by their weight of sugar.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'goods_traders'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
dispersion = rrb.TimeSeriesView(
    origin='/reporters/exchange.dispersion',
    name='spread of log price',
    axis_x=rrb.TimeAxis(view_range=whole_run),
)
grid = rrb.Spatial2DView(origin='/traders', name='traders')
blueprint = rrb.Blueprint(
    rrb.Horizontal(dispersion, grid, column_shares=[3, 2]),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=50)

# %%
# Prices converge while every good is conserved
# ---------------------------------------------
# Bilateral barter at the geometric mean of the marginal rates of substitution moves every pair toward equal rates,
# so the rates of all traders, and with them the prices, should converge (Epstein and Axtell 1996, Gooding 2019). The
# spread of the logarithm of the price should fall steadily, for two goods as for three, the mean utility should rise
# and level off, and the totals of the goods, which the invariants of the exchange check every step, should stay
# exactly at their first values.

fig, (spread, utility, totals) = plt.subplots(1, 3, figsize=(14, 4))
for goods in (2, 3):
    run = Simulation(create(MODEL, goods=goods, random_state=0), random_state=0).run(until=50).trajectory_
    spread.semilogy(run.times[1:], run['exchange.dispersion'][1:], label=f'{goods} goods')
spread.set_xlabel('step')
spread.set_ylabel('standard deviation of log price')
spread.set_title('The prices of the trades converge')
spread.legend()
trajectory = simulation.trajectory_
utility.plot(trajectory.times, trajectory['utility'])
utility.set_xlabel('step')
utility.set_ylabel('mean log utility')
utility.set_title('Barter raises the mean utility')
for good in ('sugar', 'spice', 'salt'):
    totals.plot(trajectory.times, trajectory[good] / trajectory[good][0] - 1, label=good)
totals.set_ylim(-1e-9, 1e-9)
totals.set_xlabel('step')
totals.set_ylabel('relative change of the total')
totals.set_title(f'Totals conserved, largest change {np.max(np.abs(trajectory["sugar"] - trajectory["sugar"][0])):.0e}')
totals.legend()
fig.tight_layout()
plt.show()
