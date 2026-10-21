"""
# Multiplicative noise

Free particles in a box `[0, L]` are kicked by noise whose amplitude grows across the box, as for a colloid whose
mobility depends on its position. Without forces the outcome depends on how the noise is read: in the Itô sense the
particles pile up where the noise is weak, in the anti-Itô sense, the one of physical particles at thermal
equilibrium, they spread evenly. The view shows eight particles in the Itô sense lingering near the quiet left wall.
"""

# %%
# The model is created by name from the catalog in a box of length 10 whose noise amplitude grows from 0.1 to 1.9,
# read in the Itô sense. The Euler-Maruyama stepper takes steps of 0.01, the walls reflect the particles at every
# frame of 0.1, and the observer saves every tenth frame with the default blueprint, whose primary view is the time
# series of the positions.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'multiplicative_noise'

system = create(MODEL, n_particles=8, L=10.0, noise='ito')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=10)
solver = EulerMaruyama(dt=0.01)
simulation = Simulation(system, solver=solver, dt=0.1, observers=[observer], random_state=0).run(until=299)

# %%
# Stationary densities of the three conventions
# ---------------------------------------------
# With the noise amplitude `sigma(x) = s0 + ds (x / L - 1/2)` the stationary density is `1 / sigma^2` in the Itô
# sense, `1 / sigma` in the Stratonovich sense, and flat in the anti-Itô sense. Pooled over 800 particles and the
# last half of the run, the histograms should follow these normalized densities.

grid = np.linspace(0, 10, 200)
sigma = 1.0 + 1.8 * (grid / 10 - 0.5)
fig, ax = plt.subplots(figsize=(7, 4))
for noise, power, color in (('ito', 2, 'tab:blue'), ('stratonovich', 1, 'tab:orange'), ('anti-ito', 0, 'tab:green')):
    system_n = create(MODEL, n_particles=800, L=10.0, noise=noise)
    run = Simulation(system_n, solver=solver, dt=0.1, random_state=1).run(until=300)
    edges, density = histogram(run.trajectory_['particles.x'][1500::20].ravel(), bins=25)
    ax.stairs(density, edges, color=color, label=f'{noise}')
    expected = sigma**-power
    ax.plot(grid, expected / np.trapezoid(expected, grid), '--', color=color, label=f'1 / sigma^{power}')
ax.set_xlabel('position x')
ax.set_yscale('log')
ax.set_ylabel('probability density')
ax.set_title('Where the particles settle depends on the noise convention')
ax.legend(ncols=3, fontsize='small')
fig.tight_layout()
plt.show()
