"""
# Ramsey optimal consumption

A planner splits the output of an economy with little capital between consumption and investment to maximise the
discounted utility of consumption. Along the optimal path both capital and consumption rise towards the modified golden
rule, and the economy saves a falling share of its output as it grows richer.
"""

# %%
# The system is built from its class with the Cobb-Douglas output `k^0.3`, depreciation 0.05, the discount
# rate 0.03, and the initial capital 1. Its tuned collocation solver solves the problem over 40 years.
# The default blueprint shows capital and consumption in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.macroeconomics import OptimalConsumptionSystem

SYSTEM = 'optimal_consumption'

system = OptimalConsumptionSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer])

# %%
# The Euler equation
# ------------------
# Pontryagin's principle gives the Euler equation `c' / c = alpha k^(alpha - 1) - delta - rho`, so the growth rate of
# consumption, computed from the solution, should match the marginal product of capital net of depreciation and the
# discount rate. It falls to 0 as capital approaches the golden rule `k* = (alpha / (delta + rho))^(1 / (1 - alpha))`.

times, capital, consumption = trajectory.times, trajectory.get_values('economy.k'), trajectory.get_values('economy.c')
growth = np.gradient(np.log(consumption), times)
euler = 0.3 * capital ** (0.3 - 1) - 0.05 - 0.03
golden = (0.3 / 0.08) ** (1 / 0.7)
fig, (path, rates) = plt.subplots(1, 2, figsize=(10, 4))
path.plot(times, capital, label='capital k')
path.plot(times, consumption, label='consumption c')
path.axhline(golden, color='black', linestyle='--', linewidth=0.8, label=f'golden rule k* = {golden:.2f}')
path.set_xlabel('time (years)')
path.set_ylabel('per worker')
path.set_title('Capital and consumption rise together')
path.legend()
rates.plot(times, growth, label="c' / c from the solution")
rates.plot(times, euler, '--', label='alpha k^(alpha - 1) - delta - rho')
rates.set_xlabel('time (years)')
rates.set_ylabel('growth rate of consumption')
rates.set_title('The solution follows the Euler equation')
rates.legend()
fig.tight_layout()
plt.show()
