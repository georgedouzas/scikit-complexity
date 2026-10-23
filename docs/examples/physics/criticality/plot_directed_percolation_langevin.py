"""
# Directed percolation Langevin field

An active density fed by an energy field sits just above its absorbing-state transition. Its multiplicative noise
vanishes where the density does, so activity survives in shifting patches that spread, merge, and die out. The view
shows the active density on a 64 x 64 lattice, small bright patches of activity scattered over dark regions near the
absorbing state.
"""

# %%
# The model is created by name from the catalog with the energy `nu = 1.05`, just above the threshold `nu_c = 1` of
# the growth rate `a + b nu` with `a = -1` and `b = 1`, and with the noise amplitude `e = 0.5`. Without drive and
# dissipation the energy stays fixed, so it is the control parameter. The Euler-Maruyama solver of the catalog
# integrates the noise. The default blueprint puts the view of the active density `rho` first, with the energy beside
# it.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'directed_percolation_langevin'

system = create(MODEL, energy=1.05, e=0.5)
blueprint = default_blueprint(system, primary='neurons/rho')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1.0, solver=EulerMaruyama(dt=0.01), observers=[observer], random_state=0)
simulation.run(until=100)

# %%
# The absorbing transition
# ------------------------
# In mean field the stationary density is `(a + b nu) / c`, so it is zero up to `nu_c = 1` and grows linearly above
# it (Tian et al., Eqs. 5-10). The stationary density of short runs on a 16 x 16 lattice should vanish below the
# threshold and follow the mean-field line above it, rounded near the threshold by the fluctuations.

energies = np.linspace(0.6, 1.8, 13)
densities = []
for energy in energies:
    run = Simulation(create(MODEL, shape=(16, 16), energy=energy), dt=1.0, solver=EulerMaruyama(dt=0.01))
    densities.append(np.asarray(run.run(until=40).trajectory_['neurons.rho'])[-10:].mean())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(energies, densities, 'o', label='Langevin field, 16 x 16')
ax.plot(energies, np.maximum(energies - 1.0, 0.0), 'k--', label='mean field (a + b nu) / c')
ax.axvline(1.0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('energy nu')
ax.set_ylabel('stationary active density')
ax.set_title('Activity vanishes below nu_c = 1')
ax.legend()
fig.tight_layout()
plt.show()
