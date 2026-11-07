"""
# Hasselmann's stochastic climate

Fast, white weather noise drives a slow ocean that forgets it only gradually: the temperature anomaly wanders as red
noise, with long excursions above and below zero instead of the rapid flicker of the weather.
"""

# %%
# The system is created by name from the catalog with the damping `lam = 0.5` and the noise amplitude `sigma = 0.3`, and
# is integrated with the Euler-Maruyama scheme. The default blueprint shows the anomaly in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import correlation, distribution
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'hasselmann'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = (
    Simulation(
        system,
        dt=0.1,
        solver=EulerMaruyamaSolver(dt=0.01),
        observers=[observer],
        random_state=0,
    )
    .prepare()
    .run(
        until=100,
    )
)

# %%
# The Ornstein-Uhlenbeck statistics
# ---------------------------------
# The anomaly is an Ornstein-Uhlenbeck process, so in a long run its autocorrelation should decay as `exp(-lam t)` and
# its values should follow a normal distribution of variance `sigma^2 / (2 lam)`.

lam, sigma = 0.5, 0.3
run = (
    Simulation(create_system(SYSTEM), dt=0.1, solver=EulerMaruyamaSolver(dt=0.01), random_state=1)
    .prepare()
    .run(until=2000)
    .trajectory_
)
anomaly = run.values('ocean.T')[100:]
lags = np.arange(80) * 0.1
histogram = distribution(anomaly, kind='histogram', bins=40)
edges, density = histogram.edges, histogram.density
variance = sigma**2 / (2 * lam)
grid = np.linspace(edges[0], edges[-1], 200)
fig, (memory, distribution_axes) = plt.subplots(1, 2, figsize=(10, 4))
autocorrelation = correlation(anomaly, max_lag=79).correlation[0]
memory.plot(lags, autocorrelation, label='simulation')
memory.plot(lags, np.exp(-lam * lags), color='black', linestyle='--', label='exp(-lam t)')
memory.set_xlabel('lag')
memory.set_ylabel('autocorrelation')
memory.set_title('The ocean forgets on the time 1 / lam')
memory.legend()
distribution_axes.stairs(density, edges, fill=True, alpha=0.5, label='simulation')
distribution_axes.plot(grid, np.exp(-(grid**2) / (2 * variance)) / np.sqrt(2 * np.pi * variance), color='black')
distribution_axes.set_xlabel('temperature anomaly T')
distribution_axes.set_ylabel('probability density')
distribution_axes.set_title('Stationary variance sigma^2 / (2 lam)')
fig.tight_layout()
plt.show()
