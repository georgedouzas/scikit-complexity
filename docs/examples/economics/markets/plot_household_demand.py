"""
# Household demand

A thousand households with unequal budgets buy food. With Cobb-Douglas preferences each spends a fixed share of its
budget on food whatever the price, so the market demand curve is a hyperbola. With budget shares and own-price
elasticities that differ by income, as in practical demand modelling, a rise in the price changes the expenditure of
the rich and the poor differently.
"""

# %%
# The budgets follow a shifted exponential law with a mean of 100, and the price of food rises by 10 percent a step
# from 1, for 10 steps. The default blueprint shows the total demand and expenditure of the Cobb-Douglas households.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'household_demand'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
utility = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run(until=10).trajectory_
practical = create_simulation(SYSTEM, demand='elasticity').prepare().run(until=1).trajectory_

# %%
# Demand curves and who pays for a price rise
# -------------------------------------------
# The Cobb-Douglas demand curve should follow `sum(alpha m) / p`, with a constant expenditure. With the elasticities
# of Box 3.6, a price rise of 10 percent should lower the demand by about 4 percent and raise the expenditure by about
# 5 percent, by more in the richer quintiles, whose elasticities are smaller.

prices = utility.values('households.price')[:, 0]
budgets = practical.values('households.budget')[0]
change = practical.values('households.expenditure')[1] / practical.values('households.expenditure')[0] - 1
quintiles = np.minimum((np.argsort(np.argsort(budgets)) * 5) // len(budgets), 4)
fig, (curve, quintile) = plt.subplots(1, 2, figsize=(10, 4))
curve.plot(utility.values('total_demand'), prices, 'o-', label='simulation')
curve.plot(utility.values('total_demand')[0] / prices, prices, 'k--', lw=0.8, label='sum(alpha m) / p')
curve.set_xlabel('total demand for food')
curve.set_ylabel('price of food')
curve.set_title('Cobb-Douglas demand curve')
curve.legend()
quintile.bar(range(1, 6), [100 * change[quintiles == q].mean() for q in range(5)])
quintile.set_xlabel('budget quintile, from the poorest')
quintile.set_ylabel('change of expenditure on food (%)')
quintile.set_title('A price rise of 10 percent with elasticities')
fig.tight_layout()
plt.show()
