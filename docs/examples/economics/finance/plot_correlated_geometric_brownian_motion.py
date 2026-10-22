"""
# Correlated geometric Brownian motions

Two asset prices follow geometric Brownian motions driven by the two processes of one Wiener element, whose
covariance gives them the correlation `rho = 0.7`. The prices wander apart on their own drifts and volatilities, but
they rise and fall together, and their log returns are correlated with `rho`.
"""

# %%
# The model is created by name from the catalog with prices of 100, drifts of 5 and 8 percent, and volatilities of 20
# and 30 percent a year. It is simulated for three years in frames of 0.01 with the Euler-Maruyama scheme. The default
# blueprint shows both prices in one time series view, and the Wiener element, which has no states, draws no view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'correlated_geometric_brownian_motion'
RHO = 0.7

system = create(MODEL, rho=RHO)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
solver = EulerMaruyama(dt=0.001)
simulation = Simulation(system, dt=0.01, solver=solver, observers=[observer], random_state=0).run(until=3)

# %%
# Correlated log returns
# ----------------------
# Over one frame the log return of each asset is its volatility times its Wiener increment plus a small drift, so the
# sample correlation of the two series of log returns should be close to `rho`, and their scatter should be an ellipse
# tilted along the diagonal.

trajectory = simulation.trajectory_
first, second = (np.diff(np.log(trajectory[f'{label}.price'])) for label in ('first', 'second'))
correlation = np.corrcoef(first, second)[0, 1]
fig, (paths, returns) = plt.subplots(1, 2, figsize=(10, 4))
paths.plot(trajectory.times, trajectory['first.price'], label='first')
paths.plot(trajectory.times, trajectory['second.price'], label='second')
paths.set_xlabel('time (years)')
paths.set_ylabel('price')
paths.set_title('Prices that move together')
paths.legend()
returns.scatter(first, second, s=6, alpha=0.6)
returns.set_xlabel('log return of the first asset')
returns.set_ylabel('log return of the second asset')
returns.set_title(f'Sample correlation {correlation:.2f}, rho = {RHO}')
fig.tight_layout()
plt.show()
