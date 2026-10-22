"""
# Barter

Prisoners of war with random Red Cross parcels of chocolate and cigarettes barter in random pairs, and the price of
cigarettes in chocolate, scattered at first, settles on one value. Two barterers alone trade to the contract curve of
their Edgeworth box, where neither can gain without the other losing.
"""

# %%
# A hundred prisoners hold 1 to 9 bars of chocolate and 2 to 8 packs of cigarettes and weigh chocolate from 0.3 to
# 0.6 in their Cobb-Douglas tastes. Every step a matching law pairs them at random, and every pair trades small
# quantities at the geometric mean of their marginal rates of substitution while both gain. The blueprint shows the
# mean price of cigarettes in chocolate over the trades of every step, which settles as the trades go on.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create
from skcomplex.models.economics.markets import contract_curve

MODEL = 'barter'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters/exchange.price',
        name='price of cigarettes in chocolate',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, n_barterers=100, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=30)

# %%
# The contract curve of the Edgeworth box
# ---------------------------------------
# Two barterers with weights of chocolate `a = 0.6` and `b = 0.3` share 10 bars and 10 packs. Barter stops where their
# marginal rates of substitution are equal, so every final allocation should lie on the contract curve
# `y = b (1 - a) x Y / (a (1 - b) (X - x) + b (1 - a) x)`, within the quantity of one trade, whatever the endowment.
# Among the prisoners the spread of the logarithm of the price should fall toward zero.

A, B = 0.6, 0.3
fig, (box, spread) = plt.subplots(1, 2, figsize=(11, 4.5))
chocolate = np.linspace(0.05, 9.95, 200)
box.plot(chocolate, contract_curve(chocolate, A, B, 10.0, 10.0), color='black', label='contract curve')
rng = np.random.default_rng(1)
for index in range(8):
    x, y = rng.uniform(1, 9, 2)
    edgeworth = create(MODEL, chocolate=(x, 10 - x), cigarettes=(y, 10 - y), weights=(A, B))
    final = Simulation(edgeworth, random_state=0).run(until=2).trajectory_.agents('barterers', 2).values
    label = ('endowment', 'final allocation') if index == 0 else (None, None)
    box.plot([x, final['chocolate'][0]], [y, final['cigarettes'][0]], color='grey', lw=0.8)
    box.scatter(x, y, color='C0', label=label[0])
    box.scatter(final['chocolate'][0], final['cigarettes'][0], color='C3', label=label[1])
box.set_xlim(0, 10)
box.set_ylim(0, 10)
box.set_xlabel('chocolate of the first barterer')
box.set_ylabel('cigarettes of the first barterer')
box.set_title('Barter ends on the contract curve')
box.legend()
dispersion = simulation.trajectory_['exchange.dispersion']
spread.semilogy(simulation.trajectory_.times, dispersion, marker='.')
spread.set_xlabel('step')
spread.set_ylabel('standard deviation of log price')
spread.set_title('The prisoners settle on one price')
fig.tight_layout()
plt.show()
