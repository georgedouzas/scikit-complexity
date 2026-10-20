"""
# Rice milling system

A rice mill buys paddy from the farms, mills it into rice, husk, and bran, and sells them. The harvest is seasonal,
and the milling capacity follows the milling rate only with a delay, so paddy piles up after every harvest and the
mill idles in the lean season.
"""

# %%
# The mill procures 2000 tonnes of paddy per month on average, with a yearly cycle of relative amplitude 0.8, and runs
# for ten years. The blueprint shows the paddy stock, the rice inventory, and the milling capacity.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'rice_milling'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/mill/paddy', name='paddy'),
        rrb.TimeSeriesView(origin='/mill/rice', name='rice'),
        rrb.TimeSeriesView(origin='/mill/capacity', name='milling capacity'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(create(MODEL), dt=0.25, observers=[observer]).run(120)

# %%
# Capacity lags the harvest
# -------------------------
# The milling rate is the smaller of the paddy over the milling time and the capacity. After a harvest the capacity
# binds and paddy accumulates, while before the next harvest the paddy runs low and the capacity idles. Grain is
# conserved, so the milled paddy splits into rice, husk, and bran in fixed shares.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (rates, products) = plt.subplots(1, 2, figsize=(10, 4))
rates.plot(times, trajectory['mill.paddy_purchases'], label='paddy purchases')
rates.plot(times, trajectory['mill.milling'], label='milling')
rates.plot(times, trajectory['mill.capacity'], '--', label='capacity')
rates.set_xlabel('months')
rates.set_ylabel('tonnes of paddy per month')
rates.set_title('Seasonal procurement and milling')
rates.legend()
for path, label in [('mill.milled_rice', 'rice'), ('mill.husk_output', 'husk'), ('mill.bran_output', 'bran')]:
    products.plot(times, trajectory[path], label=label)
products.set_xlabel('months')
products.set_ylabel('tonnes per month')
products.set_title('Products of the milling')
products.legend()
fig.tight_layout()
plt.show()
