"""
# Ramsey optimal consumption

A planner splits the output of an economy with little capital between consumption and investment to maximise the
discounted utility of consumption. Along the optimal path both capital and consumption rise towards the modified golden
rule, and the economy saves a falling share of its output as it grows richer.
"""

# %%
# The model is created by name from the catalog with the Cobb-Douglas output `k^0.3`, depreciation 0.05, the discount
# rate 0.03, and the initial capital 1. The collocation solver of its catalog entry solves the problem over 40 years.
# The default blueprint shows capital and consumption in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'optimal_consumption'

system = create(MODEL)
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=0.5,
    solver=entry.defaults['solver'],
    observers=[observer],
    observables=entry.observables_of(system),
).run(until=40)

# %%
# The Euler equation
# ------------------
# Pontryagin's principle gives the Euler equation `c' / c = alpha k^(alpha - 1) - delta - rho`, so the growth rate of
# consumption, computed from the solution, should match the marginal product of capital net of depreciation and the
# discount rate. It falls to 0 as capital approaches the golden rule `k* = (alpha / (delta + rho))^(1 / (1 - alpha))`.

trajectory = simulation.trajectory_
times, capital, consumption = trajectory.times, trajectory['economy.k'], trajectory['economy.c']
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
