"""
# Hasselmann's stochastic climate

Fast, white weather noise drives a slow ocean that forgets it only gradually: the temperature anomaly wanders as red
noise, with long excursions above and below zero instead of the rapid flicker of the weather.
"""

# %%
# The model is created by name from the catalog with the damping `lam = 0.5` and the noise amplitude `sigma = 0.3`, and
# is integrated with the Euler-Maruyama scheme. The default blueprint shows the anomaly in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import CorrelationAnalysis, DistributionAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hasselmann'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = (
    Simulation(
        system,
        dt=0.1,
        solver=EulerMaruyamaSolver(dt=0.01),
        observers=[observer],
        random_state=0,
    )
    .check()
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
    Simulation(create(MODEL), dt=0.1, solver=EulerMaruyamaSolver(dt=0.01), random_state=1)
    .check()
    .run(until=2000)
    .trajectory_
)
anomaly = run.values('ocean.T')[100:]
lags = np.arange(80) * 0.1
histogram = DistributionAnalysis('histogram', bins=40).check().prepare(anomaly).run()
edges, density = histogram.edges_, histogram.density_
variance = sigma**2 / (2 * lam)
grid = np.linspace(edges[0], edges[-1], 200)
fig, (memory, distribution) = plt.subplots(1, 2, figsize=(10, 4))
correlation = CorrelationAnalysis(max_lag=79).check().prepare(anomaly).run().correlation_[0]
memory.plot(lags, correlation, label='simulation')
memory.plot(lags, np.exp(-lam * lags), color='black', linestyle='--', label='exp(-lam t)')
memory.set_xlabel('lag')
memory.set_ylabel('autocorrelation')
memory.set_title('The ocean forgets on the time 1 / lam')
memory.legend()
distribution.stairs(density, edges, fill=True, alpha=0.5, label='simulation')
distribution.plot(grid, np.exp(-(grid**2) / (2 * variance)) / np.sqrt(2 * np.pi * variance), color='black')
distribution.set_xlabel('temperature anomaly T')
distribution.set_ylabel('probability density')
distribution.set_title('Stationary variance sigma^2 / (2 lam)')
fig.tight_layout()
plt.show()
