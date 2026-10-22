"""
# FHP lattice gas

Particles hop along the six directions of a triangular lattice and collide at the sites, conserving their number and
their momentum. A dense disc in a uniform gas spreads as a ring of sound and then diffuses, so the gas behaves as a
fluid at large scales. The view shows the number of particles of every hexagonal cell at the end of the run.
"""

# %%
# The model is created by name from the catalog on a periodic 64 x 64 triangular lattice, where every channel is
# occupied with probability 0.2, or 0.6 inside a disc at the centre. The observer saves every step with the default
# blueprint, whose primary view shows the particle number of every cell.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.space import LatticeSpace
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fhp_lattice_gas'

system = create(MODEL, shape=(64, 64))
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=120)

# %%
# Conservation and the spreading disc
# -----------------------------------
# Collisions keep the number of particles and the momentum `sum_k n_k c_k` of every site, and propagation only moves
# them, so both totals should stay at their initial values at every step. The excess density of the disc should leave
# the centre and spread over the lattice, so the mean particle number within the disc should fall towards the mean of
# the whole gas.

trajectory = simulation.trajectory_
channels = np.stack([trajectory[f'gas.n{direction}'].astype(float) for direction in range(6)])
angles = np.arange(6) * np.pi / 3
totals = channels.sum(axis=2)
momentum = np.stack([np.cos(angles) @ totals, np.sin(angles) @ totals])
positions = LatticeSpace((64, 64), geometry='triangular').positions
disc = np.linalg.norm(positions - positions.mean(axis=0), axis=1) < 64 / 6
density = trajectory['gas.density']

fig, (conserved, spread) = plt.subplots(1, 2, figsize=(10, 4))
conserved.plot(trajectory.times, totals.sum(axis=0) - totals.sum(axis=0)[0], label='particle number')
conserved.plot(trajectory.times, momentum[0] - momentum[0, 0], label='momentum x')
conserved.plot(trajectory.times, momentum[1] - momentum[1, 0], '--', label='momentum y')
conserved.set_xlabel('time step')
conserved.set_ylabel('change from the start')
conserved.set_title('Mass and momentum are conserved')
conserved.legend()
spread.plot(trajectory.times, density[:, disc].mean(axis=1), label='inside the disc')
spread.plot(trajectory.times, density[:, ~disc].mean(axis=1), label='outside the disc')
spread.axhline(density[0].mean(), color='black', linestyle='--', label='mean of the gas')
spread.set_xlabel('time step')
spread.set_ylabel('particles per cell')
spread.set_title('The dense disc spreads into the gas')
spread.legend()
fig.tight_layout()
plt.show()
