"""
# Energy balance with ice-albedo feedback

The global temperature has two stable climates: a warm one near 286 K and a frozen one near 232 K, because ice raises
the albedo and keeps a cold planet cold. Started at 290 K, the temperature settles in the warm climate.
"""

# %%
# The example builds `EnergyBalanceSystem` with the default parameters and the initial temperature of 290 K. The
# default blueprint shows the temperature in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import EnergyBalanceSystem

SYSTEM = 'energy_balance'

system = EnergyBalanceSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, observers=[observer])

# %%
# Two climates
# ------------
# The net heating `s (1 - a(T)) - sigma epsilon T^4` vanishes at three temperatures: the warm and the frozen climates
# are stable, and the climate between them is unstable. Runs started on either side of the unstable one should end in
# different climates.

s, epsilon, sigma, albedo, contrast, T_ice, width = 1.0, 0.65, 1.6e-10, 0.5, 0.2, 263.0, 4.0
grid = np.linspace(200.0, 310.0, 1000)
heating = s * (1 - (albedo - contrast * np.tanh((grid - T_ice) / width))) - sigma * epsilon * grid**4
roots = grid[np.nonzero(np.diff(np.sign(heating)))[0]]
fig, (balance, runs) = plt.subplots(1, 2, figsize=(10, 4))
balance.plot(grid, heating)
balance.axhline(0.0, color='gray', linewidth=0.8)
balance.plot(roots[[0, 2]], [0, 0], 'o', color='black', label='stable')
balance.plot(roots[1], 0, 'o', color='white', markeredgecolor='black', label='unstable')
balance.set_xlabel('temperature T (K)')
balance.set_ylabel('dT/dt')
balance.set_title('Three climates balance the energy')
balance.legend()
for T0 in np.linspace(220.0, 300.0, 17):
    run = simulate(EnergyBalanceSystem(T=T0))
    runs.plot(run.times, run.get_values('earth.T'), color='C1' if roots[1] < T0 else 'C0', linewidth=0.8)
for root in roots:
    runs.axhline(root, color='gray', linestyle='--', linewidth=0.8)
runs.set_xlabel('time')
runs.set_ylabel('temperature T (K)')
runs.set_title('Runs split at the unstable climate')
fig.tight_layout()
plt.show()
