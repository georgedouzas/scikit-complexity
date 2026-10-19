"""
# Fully connected branching and annihilation

Above the absorbing transition `kappa = nu`, the number of particles in a volume `V = 100` fluctuates around the active
density `rho_0 = (kappa - nu) / lambda` instead of falling into the empty state, which it would only reach through a
fluctuation that takes a time growing like `exp(V phi(0))`. The time series shows the number of particles.
"""

# %%
# The model is created by name from the catalog with `kappa = 2`, `nu = lambda = 1`, and `V = 100`, so `rho_0 = 1`.
# The observer saves every frame of the run with the default blueprint, whose primary view is the time series of the
# number of particles. The Numba backend runs the jumps in a compiled loop.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'absorbing_birth_death'

system = create(MODEL, kappa=2.0, volume=100.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer], random_state=0, backend='numba').run(until=100)

# %%
# Large deviations of the density and the absorbing transition
# -------------------------------------------------------------
# In the metastable active state the density `rho = n / V` has the distribution `exp(-V phi(rho))` of Bertin's
# equation (3.44), `phi(rho) = -(rho - rho_0)(1 + ln kappa) + ((nu + lambda rho) ln(nu + lambda rho) - kappa ln
# kappa) / lambda`. The histogram of a long run should match it. Across the transition the time-averaged density of
# runs started at `rho = 1` should follow the mean-field density `max(0, (kappa - nu) / lambda)`, with zero below
# `kappa = nu` where every run is absorbed.


def large_deviation(rho, kappa, nu=1.0, lam=1.0):
    """Return the large deviation function of the density in the metastable active state."""
    rho_0 = (kappa - nu) / lam
    rate = nu + lam * rho
    return -(rho - rho_0) * (1 + np.log(kappa)) + (rate * np.log(rate) - kappa * np.log(kappa)) / lam


long = Simulation(create(MODEL, kappa=2.0, volume=100.0), dt=0.5, random_state=1, backend='numba').run(until=4000)
density = long.trajectory_['particles.n'][100:] / 100
kappas = np.linspace(0.6, 2.4, 10)
means = []
for kappa in kappas:
    run = Simulation(create(MODEL, kappa=kappa), dt=0.5, random_state=0, backend='numba').run(until=200)
    means.append((run.trajectory_['particles.n'][100:] / 100).mean())
grid = np.linspace(0.5, 1.5, 200)
weights = np.exp(-100 * large_deviation(grid, 2.0))
fig, (distribution, transition) = plt.subplots(1, 2, figsize=(10, 4))
distribution.hist(density, bins=np.arange(0.495, 1.5, 0.01), density=True, alpha=0.6, label='simulation')
distribution.plot(grid, weights / np.trapezoid(weights, grid), color='black', label='exp(-V phi(rho))')
distribution.set_xlabel('density rho')
distribution.set_ylabel('probability density')
distribution.set_title('Fluctuations of the active state, V = 100')
distribution.legend()
transition.plot(kappas, means, 'o', label='simulation')
transition.plot(kappas, np.maximum(0, kappas - 1.0), color='black', label='(kappa - nu) / lambda')
transition.set_xlabel('branching rate kappa / nu')
transition.set_ylabel('mean density')
transition.set_title('Absorbing transition at kappa = nu')
transition.legend()
fig.tight_layout()
plt.show()
