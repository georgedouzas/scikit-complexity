"""
# Contact process

Infected sites recover at rate one and infect each neighbour at rate `lambda / 4`. Above the critical rate
`lambda_c = 1.65` of the square lattice the infection survives forever at a steady density, below it dies out into
the absorbing healthy state. Close above `lambda_c`, as here, the infected sites (yellow) gather into sparse,
irregular clusters that keep splitting, merging, and dying out. The view shows a 128 x 128 lattice at `lambda = 1.75`.
"""

# %%
# The model is created by name from the catalog with half of the sites infected at the start and a time slice
# `tau = 0.1` per step. The observer saves every second step with the default blueprint, whose primary view is the
# image of the infections.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'contact_process'

system = create(MODEL, shape=(128, 128), rate=1.75, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=500)

# %%
# Absorbing-state transition
# --------------------------
# The steady density of infected sites on a 64 x 64 lattice should vanish below `lambda_c = 1.65` and grow
# continuously above it. The mean-field density `1 - 1 / lambda` ignores the correlations between neighbours and
# puts the transition at `lambda = 1`, so the simulation should lie below it, with a transition shifted to the
# right.

rates = np.linspace(1.0, 4.0, 16)
densities = []
for rate in rates:
    run = Simulation(create(MODEL, shape=(64, 64), rate=rate, random_state=1), dt=1, random_state=1).run(until=1500)
    densities.append(run.trajectory_['lattice.infected'][1000:].mean())
grid = np.linspace(1.0, 4.0, 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rates, densities, 'o-', label='simulation, 64 x 64')
ax.plot(grid, 1 - 1 / grid, color='black', linestyle=':', label='mean field, 1 - 1 / lambda')
ax.axvline(1.65, color='gray', linestyle='--', label='lambda_c = 1.65')
ax.set_xlabel('infection rate lambda')
ax.set_ylabel('steady density of infected sites')
ax.set_title('The infection survives above lambda_c')
ax.legend()
fig.tight_layout()
plt.show()
