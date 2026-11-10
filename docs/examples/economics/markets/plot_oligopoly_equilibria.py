"""
# Static oligopoly equilibria

Firms with loyal customers and switchers have no equilibrium in pure prices: each would undercut the other down to where
selling only to its loyal customers at the top price pays more. The symmetric equilibrium mixes prices with the
distribution `F(P)` that makes a firm indifferent among them, at the profit `alpha`. Two firms that learn by fictitious
play, each best responding to the prices of its rival so far, charge prices distributed near it.
"""

# %%
# The static system solves the Cournot best responses and the distribution on a grid of prices in one frame.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import sweep
from skcomplex.systems.economics.markets import OligopolyEquilibriaSystem

SYSTEM = 'oligopoly_equilibria'

system = OligopolyEquilibriaSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=1.0, observers=[observer])

# %%
# The distribution of prices
# --------------------------
# A `sweep` varies the number of firms, whose rivals drive prices up, and a run of 2000 rounds of fictitious play
# gives the empirical distribution of the duopoly.

bounds = {'n_firms': [2, 3], 'alpha': [0.2]}
result = sweep(OligopolyEquilibriaSystem(), bounds, until=1.0, outputs=['prices.price', 'prices.distribution'])
prices, distributions = result.get_series('prices.price')[:, 0], result.get_series('prices.distribution')[:, 0]
played = simulate(OligopolyEquilibriaSystem(dynamics='fictitious-play'), until=2000, random_state=0)
charged = np.sort(np.concatenate([played.get_values('lower_price')[1:], played.get_values('higher_price')[1:]]))
grid = np.linspace(1 / 3, 1.0, 100)
fig, (static, learned) = plt.subplots(1, 2, figsize=(10, 4))
for n_firms, price, distribution in zip((2, 3), prices, distributions, strict=True):
    static.plot(price, distribution, 'o-', label=f'm = {n_firms}')
static.set_xlabel('price P')
static.set_ylabel('F(P)')
static.set_title('Equilibrium price distributions')
static.legend()
learned.plot(charged, np.arange(1, charged.size + 1) / charged.size, drawstyle='steps-post', label='fictitious play')
learned.plot(grid, 1 - 0.25 * (1 - grid) / (0.5 * grid), color='black', label='F(P), alpha = 0.25')
learned.set_xlabel('price P')
learned.set_ylabel('share of prices charged up to P')
learned.set_title('Learning approaches the mixed equilibrium')
learned.legend()
fig.tight_layout()
plt.show()
