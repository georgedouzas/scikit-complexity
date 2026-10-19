"""
# Sherrington-Kirkpatrick spin glass

Below the glass temperature `T_c = J`, 200 spins with random Gaussian couplings freeze in random directions. The
space-time image shows the spins drawn in a row, one row per frame with the earliest at the top: after a short
transient most columns keep one color, a frozen pattern of both colors with no net magnetization.
"""

# %%
# The model is created by name from the catalog with 200 spins at `T = 0.3 J`, starting from random spins. One unit
# of time is one attempted flip per spin. The observer saves every frame with the default blueprint, whose primary
# view is the space-time image of the spins.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sherrington_kirkpatrick'

system = create(MODEL, n_spins=200, temperature=0.3)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer], random_state=0).run(until=100)

# %%
# Energy and freezing against temperature
# ---------------------------------------
# Above `T_c = J` the replica-symmetric theory gives the energy per spin `-J^2 / (2 T)` and no frozen order. Below
# `T_c` the energy bends away from this line toward the ground-state energy of Parisi's solution, about `-0.763 J`,
# and the Edwards-Anderson order parameter `q = mean_i <s_i>^2`, measured from time averages of the spins over the
# second half of each run, rises from near zero. The simulated energies of the same 200-spin sample should follow
# `-J^2 / (2 T)` above `T_c`, and `q` should grow below it.

couplings = system.elements[0].values['couplings']
temperatures = np.linspace(0.2, 2.0, 7)
energies, frozen = [], []
for temperature in temperatures:
    sample = create(MODEL, n_spins=200, temperature=temperature)
    spins = Simulation(sample, dt=1.0, random_state=0).run(until=100).trajectory_['spins.spin'][50:]
    energies.append(-0.5 * np.einsum('ti,ij,tj->t', spins, couplings, spins).mean() / 200)
    frozen.append(np.mean(spins.mean(axis=0) ** 2))
grid = np.linspace(1.0, 2.1, 50)
fig, (energy, order) = plt.subplots(1, 2, figsize=(10, 4))
energy.plot(temperatures, energies, 'o', label='simulation, N = 200')
energy.plot(grid, -1 / (2 * grid), color='black', label='-J^2 / (2 T) above T_c')
energy.axhline(-0.7633, color='gray', linestyle='--', label='Parisi ground state')
energy.set_xlabel('temperature T / J')
energy.set_ylabel('energy per spin')
energy.set_title('Energy follows -1 / (2 T) above T_c')
energy.legend()
order.plot(temperatures, frozen, 'o-')
order.axvline(1.0, color='black', linestyle='--', label='T_c = J')
order.set_xlabel('temperature T / J')
order.set_ylabel('q = mean of <s_i>^2')
order.set_title('Spins freeze below T_c')
order.legend()
fig.tight_layout()
plt.show()
