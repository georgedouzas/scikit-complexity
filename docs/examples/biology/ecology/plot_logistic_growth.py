"""
# Logistic growth

The population grows almost exponentially while it is small, slows down as it fills the habitat, and levels off at
its carrying capacity of 100 along an S-shaped curve.
"""

# %%
# The example builds `LogisticGrowthSystem` and runs it from a population of two until it has saturated. The observer
# saves every frame with the default blueprint, whose primary view is the time series of the population.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

SYSTEM = 'logistic_growth'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(LogisticGrowthSystem(n=2.0, r=0.5, K=100.0), observers=[observer])

# %%
# Closed-form solution and growth rate
# ------------------------------------
# The logistic law `dn/dt = r n (1 - n / K)` has the solution `n(t) = K / (1 + (K / n0 - 1) exp(-r t))`. The
# simulated population should lie on it, and its growth rate `dn/dt` should be a parabola in `n` that peaks at `K / 2`.

times, population = trajectory.times, trajectory.get_values('population.x')
r, K, n0 = 0.5, 100.0, 2.0
exact = K / (1 + (K / n0 - 1) * np.exp(-r * times))
fig, (curve, rate) = plt.subplots(1, 2, figsize=(10, 4))
curve.plot(times, population, label='simulated')
curve.plot(times, exact, '--', color='black', label='closed form')
curve.axhline(K, color='grey', linestyle=':', label='carrying capacity')
curve.set_xlabel('time')
curve.set_ylabel('population')
curve.set_title('The population follows the logistic curve')
curve.legend()
rate.plot(population[1:], np.diff(population) / np.diff(times), label='simulated')
grid = np.linspace(0, K, 100)
rate.plot(grid, r * grid * (1 - grid / K), '--', color='black', label='r n (1 - n / K)')
rate.axvline(K / 2, color='grey', linestyle=':', label='K / 2')
rate.set_xlabel('population')
rate.set_ylabel('growth rate dn/dt')
rate.set_title('Growth is fastest at half the capacity')
rate.legend()
fig.tight_layout()
plt.show()
