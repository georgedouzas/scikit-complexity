"""
# Hasselmann's stochastic climate

Fast, white weather noise drives a slow ocean that forgets it only gradually: the temperature anomaly wanders as red
noise, with long excursions above and below zero instead of the rapid flicker of the weather.
"""

# %%
# The system is built from its class with the damping `lam = 0.5` and the noise amplitude `sigma = 0.3`, and
# is integrated with the Euler-Maruyama scheme. The default blueprint shows the anomaly in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import correlation, distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import HasselmannSystem

SYSTEM = 'hasselmann'

system = HasselmannSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# The Ornstein-Uhlenbeck statistics
# ---------------------------------
# The anomaly is an Ornstein-Uhlenbeck process, so in a long run its autocorrelation should decay as `exp(-lam t)` and
# its values should follow a normal distribution of variance `sigma^2 / (2 lam)`.

lam, sigma = 0.5, 0.3
run = simulate(HasselmannSystem(), until=2000, random_state=1)
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
