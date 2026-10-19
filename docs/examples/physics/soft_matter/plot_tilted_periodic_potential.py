"""
# Tilted periodic potential

The particles leave the origin and drift to the right down the tilted periodic potential, hopping from well to well,
while thermal noise spreads them out, so the cloud moves at a steady mean velocity and widens.
"""

# %%
# The model is created by name from the catalog with 20 particles in the potential `0.5 cos(2 pi x) - f x` at
# `kT = 0.5`, with the force `f = 1`, and simulated for 50 time units. The observer saves every frame with the default
# blueprint, whose primary view holds the positions of the particles.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tilted_periodic_potential'

system = create(MODEL, n_particles=20, f=1.0, U0=0.5, L=1.0, kT=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.25, solver=EulerMaruyama(dt=0.005), observers=[observer], random_state=0)
simulation.run(until=50)

# %%
# Mean velocity against the force
# -------------------------------
# On the periodic interval the steady probability current of the Fokker-Planck equation gives the mean velocity
# `v = L (1 - exp(-f L / kT)) / int_0^L dx int_0^L dy exp((V(x) - V(x - y)) / kT) kT / gamma`, with
# `V(x) = U0 cos(2 pi x / L) - f x`. The mean velocities measured over a range of forces should follow this curve,
# which stays far below the free drift `f / gamma` for weak forces, where the barriers hold the particles back, and
# joins it for strong forces. The positions of the run above should drift linearly in time at the predicted velocity.


def tilted_potential(position, f, U0=0.5, L=1.0):
    return U0 * np.cos(2 * np.pi * position / L) - f * position


def exact_velocity(f, U0=0.5, L=1.0, kT=0.5, gamma=1.0):
    x = np.linspace(0.0, L, 400, endpoint=False)
    first, second = np.meshgrid(x, x, indexing='ij')
    difference = tilted_potential(first, f, U0, L) - tilted_potential(first - second, f, U0, L)
    integral = np.exp(difference / kT).mean() * L**2
    return L * (1 - np.exp(-f * L / kT)) / integral * kT / gamma


forces = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
velocities = []
for f in forces:
    run = Simulation(create(MODEL, n_particles=300, f=f), dt=1.0, solver=EulerMaruyama(dt=0.005), random_state=1)
    positions = run.run(until=30).trajectory_['particles.x'][..., 0]
    velocities.append((positions[-1] - positions[5]).mean() / 25)

times = simulation.trajectory_.times
mean_position = simulation.trajectory_['particles.x'][..., 0].mean(axis=1)

fig, (current, drift) = plt.subplots(1, 2, figsize=(10, 4))
grid = np.linspace(0.05, 4.0, 60)
current.plot(grid, [exact_velocity(f) for f in grid], color='black', label='exact current')
current.plot(grid, grid, color='grey', linestyle='--', label='free drift f / gamma')
current.plot(forces, velocities, 'o', label='simulated')
current.set_xlabel('force f')
current.set_ylabel('mean velocity')
current.set_title('A tilt drives a steady current')
current.legend()
drift.plot(times, mean_position, label='simulated mean position')
drift.plot(times, exact_velocity(1.0) * times, color='black', linestyle='--', label='exact velocity times t')
drift.set_xlabel('time')
drift.set_ylabel('mean position')
drift.set_title('The particles drift at a constant velocity')
drift.legend()
fig.tight_layout()
plt.show()
