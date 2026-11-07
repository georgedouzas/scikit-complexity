"""
# Toy Trader

Four hundred traders on a torus start with 100 money and 5 toys. Each buys the cheapest toy it can see and sellers
raise their price when their toys run short and lower it when they pile up. Supply and demand never change, yet the
price falls when traders see only their neighbours and rises when they see the whole market, and money concentrates.
"""

# %%
# Traders see within one unit and move one unit at random every iteration, for 300 iterations. The default blueprint
# shows the average price, the trades, and the Gini coefficients of money and assets.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.markets import ToyTraderSystem

SYSTEM = 'toy_trader'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulate(ToyTraderSystem(), observers=[observer])

# %%
# Market sight and evolution
# --------------------------
# As in Figs. 10.1 to 10.3 of Gooding, the price should fall with a sight of 1 and rise with a sight that covers the
# torus. With mutating pricing factors and a fitness test of assets, the poor copying the prices of the rich, the price
# should inflate like an asset market (Fig. 10.9). The Gini coefficient of money should approach 0.5.

scenarios = {
    'sight 1': {},
    'sight 4': {'sight': 4.0},
    'sight 40': {'sight': 40.0},
    'sight 1, asset fitness': {'mutation': 1 / 180, 'fitness': 'assets'},
}
fig, (price, gini) = plt.subplots(1, 2, figsize=(10, 4))
for name, settings in scenarios.items():
    trajectory = simulate(ToyTraderSystem(**settings))
    times = np.asarray(trajectory.times)
    price.plot(times, trajectory.values('market.average_price'), label=name)
    gini.plot(times, trajectory.values('market.money_gini'), label=name)
price.set_yscale('log')
price.set_xlabel('iteration')
price.set_ylabel('average price of a toy')
price.set_title('Price against market sight')
price.legend()
gini.axhline(0.5, color='black', linestyle=':', lw=0.8)
gini.set_xlabel('iteration')
gini.set_ylabel('Gini coefficient of money')
gini.set_title('Inequality')
fig.tight_layout()
plt.show()
