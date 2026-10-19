"""
# Edwards-Anderson spin glass

At low temperature the spins of the two-dimensional `+J` and `-J` glass settle into a frozen pattern of both colors
with no ferromagnetic domains, since every spin can only satisfy part of its random bonds. The image shows the spins
of the 48 x 48 lattice at the end of the run.
"""

# %%
# The model is created by name from the catalog on a 48 x 48 lattice at `T = 0.5 J`, starting from random spins. One
# unit of time is one sweep of Metropolis updates. The observer saves every second frame with the default blueprint,
# whose primary view is the image of the spins.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'edwards_anderson'

system = create(MODEL, size=48, temperature=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=5, observers=[observer], random_state=0).run(until=1000)

# %%
# Energy, magnetization, and freezing against temperature
# -------------------------------------------------------
# At high temperature the energy per spin of `+J` and `-J` bonds is `-2 J tanh(J / T)`, and at low temperature it
# approaches the ground-state energy, about `-1.40 J`. The magnetization stays near zero at every temperature, while
# the Edwards-Anderson parameter `q = mean_i <s_i>^2`, from time averages over the last 50 frames, grows as the spins
# freeze. The simulated energies on the same 32 x 32 sample should follow the high-temperature curve above
# `T = 2 J` and level off near `-1.40 J` below `T = 0.5 J`.

temperatures = np.linspace(0.3, 3.5, 9)
energies, magnetizations, frozen = [], [], []
for temperature in temperatures:
    sample = create(MODEL, size=32, temperature=temperature)
    values = sample.elements[0].values
    right, down = (values[name].reshape(32, 32) for name in ('J_right', 'J_down'))
    trajectory = Simulation(sample, dt=10, random_state=0).run(until=2000).trajectory_
    spins = trajectory.lattice('spins.spin')[-50:]
    bonds = spins * right * np.roll(spins, -1, axis=-1) + spins * down * np.roll(spins, -1, axis=-2)
    energies.append(-bonds.sum(axis=(1, 2)).mean() / 32**2)
    magnetizations.append(np.abs(trajectory['magnetization'][-50:]).mean())
    frozen.append(np.mean(spins.mean(axis=0) ** 2))
grid = np.linspace(1.5, 3.6, 50)
fig, (energy, order) = plt.subplots(1, 2, figsize=(10, 4))
energy.plot(temperatures, energies, 'o', label='simulation, 32 x 32')
energy.plot(grid, -2 * np.tanh(1 / grid), color='black', label='-2 J tanh(J / T)')
energy.axhline(-1.40, color='gray', linestyle='--', label='ground state')
energy.set_xlabel('temperature T / J')
energy.set_ylabel('energy per spin')
energy.set_title('Energy approaches the ground state')
energy.legend()
order.plot(temperatures, frozen, 'o-', label='q = mean of <s_i>^2')
order.plot(temperatures, magnetizations, 's-', label='mean |m|')
order.set_xlabel('temperature T / J')
order.set_ylabel('order parameter')
order.set_title('Frozen spins without magnetization')
order.legend()
fig.tight_layout()
plt.show()
